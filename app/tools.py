import os
import requests
import uuid
import re
import logging
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
from app.rag import RAGManager
from app.ocr import DeepSeekOCR
from pdf2image import convert_from_path

# Add logging
logger = logging.getLogger(__name__)

# Initialize RAG Manager
rag_manager = RAGManager()

# Configuration
TEMP_DIR = Path("data/temp")  # Use Path and separate temp folder
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
MAX_PDF_PAGES = 3  # Configurable

def clean_deepseek_output(text: str) -> str:
    """Removes DeepSeek's special tags to leave clean Markdown."""
    if not text:
        return ""
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove image URLs
    text = re.sub(r'https?://[^\s]+', '', text)  # Remove standalone URLs
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

def search_competitors(query: str, max_results: int = 3):
    """Finds URLs using Tavily."""
    logger.info(f"Searching: {query}")
    search = TavilySearchResults(max_results=max_results)
    return search.invoke(query)

def reset_knowledge_base() -> None:
    """Clears the RAG database for a new run."""
    logger.info("Resetting knowledge base")
    rag_manager.clear()
    
def process_visual_url(url: str) -> str:
    """Downloads image/PDF, runs OCR, and indexes it."""
    temp_files = []  # Track files for cleanup
    
    try:
        logger.info(f"Processing: {url}")
        
        # 1. Download with size limit
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, stream=True, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"Failed to download {url}. Status: {response.status_code}"
        
        # Check file size before downloading
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return f"File too large: {int(content_length) / 1024 / 1024:.1f}MB"
        
        # Determine file type
        is_pdf = url.lower().endswith('.pdf') or 'application/pdf' in response.headers.get('Content-Type', '')
        ext = ".pdf" if is_pdf else ".png"
        filename = f"{uuid.uuid4()}{ext}" 
        local_path = TEMP_DIR / filename
        temp_files.append(local_path)  # Track for cleanup
        
        # Download with size checking
        downloaded_size = 0
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_FILE_SIZE:  # Safety check
                        local_path.unlink()
                        return f"File exceeded size limit during download"
        
        # 2. Prepare Images
        ocr_images = []
        if is_pdf:
            logger.info(f"Converting PDF (first {MAX_PDF_PAGES} pages)")
            try:
                images = convert_from_path(
                    local_path, 
                    first_page=1, 
                    last_page=MAX_PDF_PAGES,
                    dpi=200  # Better quality
                )
                
                for i, img in enumerate(images):
                    page_filename = f"{uuid.uuid4()}_page{i+1}.jpg"
                    page_path = TEMP_DIR / page_filename
                    img.save(page_path, 'JPEG', quality=85)
                    ocr_images.append(page_path)
                    temp_files.append(page_path)  # Track for cleanup
            except Exception as pdf_err:
                return f"PDF Conversion Failed: {pdf_err}"
        else:
            ocr_images.append(local_path)
        
        # 3. Run OCR
        ocr_engine = DeepSeekOCR.get_instance()
        combined_text = ""
        
        for img_path in ocr_images:
            logger.info(f"OCR: {img_path.name}")
            raw_text = ocr_engine.process_image(str(img_path))
            clean_chunk = clean_deepseek_output(raw_text)
            combined_text += f"\n\n--- Page Break ---\n\n{clean_chunk}"
        
        # 4. Validate and Index
        if "OCR Failed" in combined_text or len(combined_text.strip()) < 50:
            return f"OCR returned insufficient text for {url}"
        
        logger.info(f"Indexing {len(combined_text)} chars")
        rag_manager.add_documents(combined_text, source_url=url)
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {temp_file}: {e}")
        
        return f"Successfully processed {url} ({len(ocr_images)} pages)"
        
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        # Cleanup on error
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        return f"Error processing {url}: {type(e).__name__}"

def query_knowledge_base(question: str, top_k: int = 20) -> str:
    """Query the RAG knowledge base."""
    logger.info(f"Querying: {question[:50]}...")
    results = rag_manager.retrieve(question, k=top_k)
    return "\n\n".join([doc.page_content for doc in results])
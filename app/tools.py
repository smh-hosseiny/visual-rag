import os
import requests
import uuid
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from app.rag import RAGManager
from app.ocr import DeepSeekOCR
from pdf2image import convert_from_path

# Initialize RAG Manager
rag_manager = RAGManager()

# Ensure temp directory exists
TEMP_DIR = "data"
os.makedirs(TEMP_DIR, exist_ok=True)

def clean_deepseek_output(text: str) -> str:
    """Removes DeepSeek's special tags (<|ref|>, <|det|>) to leave clean Markdown."""
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    return text

def search_competitors(query: str):
    """Finds URLs of competitors using Tavily."""
    # Suppress deprecation warning by just using it
    search = TavilySearchResults(max_results=3)
    return search.invoke(query)

def reset_knowledge_base():
    """Clears the RAG database for a new run."""
    rag_manager.clear()
    

def process_visual_url(url: str):
    """Downloads image/PDF, converts if needed, runs OCR, cleans text, and indexes it."""
    try:
        print(f"⬇️ Downloading: {url}")
        
        # 1. Download Content
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return f"Failed to download {url}. Status: {response.status_code}"

        # Determine if it's a PDF
        is_pdf = url.lower().endswith('.pdf') or 'application/pdf' in response.headers.get('Content-Type', '')
        ext = ".pdf" if is_pdf else ".png"
        filename = f"{uuid.uuid4()}{ext}" 
        local_path = os.path.join(TEMP_DIR, filename)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        
        # 2. Prepare Images (Scan up to 3 pages)
        ocr_images = []

        if is_pdf:
            print(f"📄 PDF detected. Converting first 3 pages...")
            try:
                # --- CHANGED: Scan pages 1-3 to catch tables/data ---
                images = convert_from_path(local_path, first_page=1, last_page=3)
                
                for i, img in enumerate(images):
                    page_filename = f"{uuid.uuid4()}_page{i+1}.jpg"
                    page_path = os.path.join(TEMP_DIR, page_filename)
                    img.save(page_path, 'JPEG')
                    ocr_images.append(page_path)
            except Exception as pdf_err:
                return f"PDF Conversion Failed: {pdf_err}"
        else:
            ocr_images.append(local_path)

        # 3. Run OCR on all pages
        ocr_engine = DeepSeekOCR.get_instance()
        combined_text = ""
        
        for img_path in ocr_images:
            print(f"👁️ Running OCR on {img_path}...")
            raw_text = ocr_engine.process_image(img_path)
            clean_chunk = clean_deepseek_output(raw_text)
            combined_text += f"\n\n--- Page Break ---\n\n{clean_chunk}"

        # 4. Filter and Index
        if "OCR Failed" in combined_text or len(combined_text.strip()) < 50:
             return f"OCR returned insufficient text for {url}"

        print(f"📝 Indexing {len(combined_text)} chars into Vector DB...")
        rag_manager.add_documents(combined_text, url)
        
        return f"Successfully processed {url} ({len(ocr_images)} pages scanned)"

    except Exception as e:
        return f"Error processing {url}: {str(e)}"

# The query function remains the same
def query_knowledge_base(question: str):
    results = rag_manager.retrieve(question)
    return "\n\n".join([doc.page_content for doc in results])
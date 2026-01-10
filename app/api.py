import shutil
import os
import uuid
import torch
import asyncio
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from app.ocr import DeepSeekOCR
from app.agent import app_graph
import json
from app.tools import reset_knowledge_base

# Setup directories
UPLOAD_DIR = "data/uploads"
REPORTS_DIR = "data/reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Global progress tracking
progress_updates = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("MarketLens Agent Starting...")
    print("=" * 60)
    print("Loading DeepSeek-OCR model...")
    DeepSeekOCR.get_instance()
    print("Model loaded successfully!")
    print(f"Server running at: http://localhost:8001")
    print(f"API docs available at: http://localhost:8001/docs")
    print("=" * 60)
    yield
    print("\n" + "=" * 60)
    print("Server shutting down...")
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
    print("Shutdown complete")
    print("=" * 60)

app = FastAPI(title="MarketLens Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    print("Health check received")
    return {"status": "online", "message": "MarketLens is ready."}


@app.post("/analyze")
async def analyze_market(
    topic: str = Form(...),
    request_id: str = Form(None),  # Accept request_id from frontend
    file: UploadFile = File(None)
):
    # Use provided request_id or generate new one
    if not request_id:
        request_id = str(uuid.uuid4())
    
    print("\n" + "=" * 60)
    print("NEW REQUEST RECEIVED")
    print(f"Request ID: {request_id}")
    print("=" * 60)
    print(f"Topic: {topic}")

    # 1. CLEAR OLD DATA FIRST!
    print("Resetting Knowledge Base for new topic...")
    reset_knowledge_base()
    
    try:
        await update_progress(request_id, "Starting analysis...")
        uploaded_file_path = None
        
        # 1. Handle File Upload
        if file:
            print(f"File uploaded: {file.filename}")
            await update_progress(request_id, f"Processing uploaded file: {file.filename}")
            
            file_extension = file.filename.split(".")[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            uploaded_file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            with open(uploaded_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(uploaded_file_path)
            print(f"File saved: {unique_filename} ({file_size} bytes)")
            await update_progress(request_id, f"File uploaded successfully ({file_size} bytes)")
        else:
            print("No file uploaded")
            await update_progress(request_id, "Searching for sources online...")
        
        # 2. Initialize Agent State
        print("\nInitializing agent...")
        await update_progress(request_id, "Initializing AI agent...")
        
        initial_state = {
            "task": topic,
            "image_urls": [uploaded_file_path] if uploaded_file_path else [],
            "report": "",
            "messages": [],
            "request_id": request_id  # Pass request_id to agent
        }
        
        # 3. Run Agent
        print("Running agent workflow...")
        await update_progress(request_id, "Researching topic and finding sources...")
        
        # Add a small delay to ensure progress is visible
        await asyncio.sleep(0.5)
        
        result = app_graph.invoke(initial_state)
        
        sources = result.get("image_urls", [])
        report = result.get("report", "Analysis failed.")
        
        print(f"\nProcessing complete!")
        print(f"  Sources processed: {len(sources)}")
        print(f"  Report length: {len(report)} characters")
        
        await update_progress(request_id, f"Found {len(sources)} sources")
        await update_progress(request_id, "Running OCR on documents...")
        await update_progress(request_id, "Analyzing data and generating report...")
        
        # 4. Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.txt"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MarketLens Analysis Report\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Sources Processed ({len(sources)}):\n")
            f.write(f"{'-' * 50}\n")
            for idx, source in enumerate(sources, 1):
                f.write(f"{idx}. {source}\n")
            f.write(f"\n{'=' * 50}\n\n")
            f.write(f"Report:\n")
            f.write(f"{'-' * 50}\n\n")
            f.write(report)
        
        print(f"Report saved: {report_filename}")
        await update_progress(request_id, "Saving report...", "completed")
        
        print("=" * 60)
        print("REQUEST COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
        return {
            "request_id": request_id,
            "topic": topic,
            "sources_processed": sources,
            "final_report": report,
            "report_file": report_filename
        }
    
    except Exception as e:
        print(f"\nERROR OCCURRED:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("=" * 60 + "\n")
        await update_progress(request_id, f"Error: {str(e)}", "failed")
        raise HTTPException(status_code=500, detail=str(e))

async def update_progress(request_id: str, step: str, status: str = "running"):
    """Update progress for a specific request"""
    if request_id not in progress_updates:
        progress_updates[request_id] = []
    progress_updates[request_id].append({
        "step": step,
        "status": status,
        "timestamp": datetime.now().isoformat()
    })
    print(f"Progress [{request_id}]: {step}")

@app.get("/progress/{request_id}")
async def get_progress(request_id: str):
    """Get progress updates for a specific request"""
    async def event_stream():
        last_index = 0
        timeout = 0
        
        while timeout < 300:  # 5 minute timeout
            if request_id in progress_updates:
                updates = progress_updates[request_id][last_index:]
                
                for update in updates:
                    yield f"data: {json.dumps(update)}\n\n"
                    last_index += 1
                    
                    # Check if completed or failed
                    if update.get("status") in ["completed", "failed"]:
                        # Clean up after sending final update
                        await asyncio.sleep(1)
                        if request_id in progress_updates:
                            del progress_updates[request_id]
                        return
            
            await asyncio.sleep(0.5)
            timeout += 0.5
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/download/{filename}")
async def download_report(filename: str):
    """Download a generated report file"""
    print(f"Download request for: {filename}")
    file_path = os.path.join(REPORTS_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="Report file not found")
    
    print(f"Serving file: {filename}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
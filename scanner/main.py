"""
FastAPI Service for Dolphin PDF Processing with Sliding Window
Compatible with OpenAI API format for file handling with prompting
"""

import asyncio
import io
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import logging
from datetime import datetime

from models.dolphin_model import DolphinModelWrapper
from models.window_processor import SlidingWindowProcessor
from models.semantic_analyzer import SemanticAnalyzer
from services.pdf_service import PDFProcessingService
from schemas.request_models import ChatCompletionRequest, ProcessPDFRequest
from schemas.response_models import ChatCompletionResponse, ProcessingResult, JobStatusResponse
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dolphin PDF Processing Service",
    description="FastAPI service for document processing with sliding window approach",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings and services
settings = Settings()
dolphin_model: Optional[DolphinModelWrapper] = None
pdf_service: Optional[PDFProcessingService] = None

# Job tracking
processing_jobs: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global dolphin_model, pdf_service
    
    logger.info("Initializing Dolphin model...")
    dolphin_model = DolphinModelWrapper(settings.model_config_path)
    
    window_processor = SlidingWindowProcessor(
        window_size=settings.default_window_size,
        overlap=settings.default_overlap
    )
    
    semantic_analyzer = SemanticAnalyzer(dolphin_model.model)
    
    pdf_service = PDFProcessingService(
        dolphin_model=dolphin_model,
        window_processor=window_processor,
        semantic_analyzer=semantic_analyzer
    )
    
    logger.info("Service initialized successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI API compatible endpoint for chat completions with file support
    Supports PDF file upload with prompting for document analysis
    """
    try:
        # Check if there's a file attached in the messages
        file_content = None
        user_prompt = ""
        
        for message in request.messages:
            if message.role == "user":
                if hasattr(message, 'content') and isinstance(message.content, list):
                    # Handle multi-modal content (text + file)
                    for content_item in message.content:
                        if content_item.type == "text":
                            user_prompt = content_item.text
                        elif content_item.type == "document" and content_item.document:
                            file_content = content_item.document
                else:
                    user_prompt = str(message.content)
        
        if not file_content:
            raise HTTPException(status_code=400, detail="No document provided for processing")
        
        # Process the document with sliding window approach
        result = await pdf_service.process_pdf_with_prompt(
            file_content=file_content,
            prompt=user_prompt,
            window_size=getattr(request, 'window_size', settings.default_window_size),
            overlap=getattr(request, 'overlap', settings.default_overlap),
            semantic_threshold=getattr(request, 'semantic_threshold', settings.default_semantic_threshold)
        )
        
        # Format response in OpenAI API format
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(datetime.utcnow().timestamp()),
            model=request.model or "dolphin-pdf-processor",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.formatted_content
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(user_prompt.split()),
                "completion_tokens": len(result.formatted_content.split()),
                "total_tokens": len(user_prompt.split()) + len(result.formatted_content.split())
            },
            processing_metadata=result.to_dict()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = "Parse and analyze this document",
    window_size: int = 2,
    overlap: int = 1,
    semantic_threshold: float = 0.8,
    output_format: str = "json"
):
    """
    Process PDF with sliding window approach
    Returns job ID for async processing
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        
        # Initialize job tracking
        processing_jobs[job_id] = {
            "status": "queued",
            "created_at": datetime.utcnow(),
            "filename": file.filename,
            "progress": 0,
            "result": None,
            "error": None
        }
        
        # Add background task
        background_tasks.add_task(
            process_pdf_background,
            job_id=job_id,
            file_content=file_content,
            prompt=prompt,
            window_size=window_size,
            overlap=overlap,
            semantic_threshold=semantic_threshold,
            output_format=output_format
        )
        
        return {"job_id": job_id, "status": "queued"}
        
    except Exception as e:
        logger.error(f"Error starting PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_pdf_background(
    job_id: str,
    file_content: bytes,
    prompt: str,
    window_size: int,
    overlap: int,
    semantic_threshold: float,
    output_format: str
):
    """Background task for PDF processing"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        
        result = await pdf_service.process_pdf_with_prompt(
            file_content=file_content,
            prompt=prompt,
            window_size=window_size,
            overlap=overlap,
            semantic_threshold=semantic_threshold
        )
        
        # Format result based on output format
        if output_format == "markdown":
            formatted_result = result.to_markdown()
        elif output_format == "html":
            formatted_result = result.to_html()
        else:
            formatted_result = result.to_dict()
        
        processing_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": formatted_result,
            "completed_at": datetime.utcnow()
        })
        
    except Exception as e:
        logger.error(f"Error processing PDF {job_id}: {str(e)}")
        processing_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow()
        })

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        created_at=job["created_at"],
        error=job.get("error")
    )

@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get processing job results"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status: {job['status']}")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "completed_at": job.get("completed_at")
    }

@app.post("/process-pages")
async def process_specific_pages(
    file: UploadFile = File(...),
    start_page: int = 1,
    end_page: Optional[int] = None,
    prompt: str = "Parse these pages",
    window_size: int = 2,
    overlap: int = 1
):
    """Process specific page ranges from PDF"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_content = await file.read()
        
        result = await pdf_service.process_pdf_pages(
            file_content=file_content,
            start_page=start_page,
            end_page=end_page,
            prompt=prompt,
            window_size=window_size,
            overlap=overlap
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error processing specific pages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    processing_jobs[job_id]["status"] = "cancelled"
    return {"message": "Job cancelled successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
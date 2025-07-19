"""
Request models for the Dolphin FastAPI service
Compatible with OpenAI API format
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import base64


class MessageContent(BaseModel):
    """Content item within a message"""
    type: str = Field(..., description="Content type: text, image_url, document")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[Dict[str, str]] = Field(None, description="Image URL with optional detail level")
    document: Optional[bytes] = Field(None, description="Document content as bytes")


class ChatMessage(BaseModel):
    """Chat message compatible with OpenAI format"""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Union[str, List[MessageContent]] = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name of the message sender")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(default="dolphin-pdf-processor", description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1, description="Nucleus sampling probability")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream partial results")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    max_tokens: Optional[int] = Field(default=4096, ge=1, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Token logit bias")
    user: Optional[str] = Field(None, description="User identifier")
    
    # Dolphin-specific parameters
    window_size: Optional[int] = Field(default=2, ge=1, le=10, description="Sliding window size")
    overlap: Optional[int] = Field(default=1, ge=0, le=5, description="Window overlap")
    semantic_threshold: Optional[float] = Field(default=0.8, ge=0, le=1, description="Semantic similarity threshold")
    merge_tables: Optional[bool] = Field(default=True, description="Merge cross-page tables")
    preserve_formulas: Optional[bool] = Field(default=True, description="Preserve formula continuity")
    output_format: Optional[str] = Field(default="text", description="Output format: text, json, markdown, html")


class ProcessPDFRequest(BaseModel):
    """Request model for PDF processing endpoint"""
    prompt: str = Field(default="Parse and analyze this document", description="Processing prompt")
    window_size: int = Field(default=2, ge=1, le=10, description="Number of pages per window")
    overlap: int = Field(default=1, ge=0, le=5, description="Overlap between windows")
    semantic_threshold: float = Field(default=0.8, ge=0, le=1, description="Similarity threshold")
    merge_tables: bool = Field(default=True, description="Merge cross-page tables")
    preserve_formulas: bool = Field(default=True, description="Preserve formula continuity")
    output_format: str = Field(default="json", description="Output format: json, markdown, html, xml")
    max_batch_size: int = Field(default=4, ge=1, le=16, description="Maximum batch size for processing")
    
    # Page range options
    start_page: Optional[int] = Field(None, ge=1, description="Start page number (1-indexed)")
    end_page: Optional[int] = Field(None, ge=1, description="End page number (1-indexed)")
    
    # Processing options
    extract_images: bool = Field(default=True, description="Extract and save images")
    extract_tables: bool = Field(default=True, description="Extract table content")
    extract_formulas: bool = Field(default=True, description="Extract mathematical formulas")
    
    # Quality options
    image_quality: str = Field(default="high", description="Image processing quality: low, medium, high")
    text_detection_confidence: float = Field(default=0.5, ge=0, le=1, description="Text detection confidence threshold")


class ProcessPagesRequest(BaseModel):
    """Request model for processing specific page ranges"""
    start_page: int = Field(..., ge=1, description="Start page number (1-indexed)")
    end_page: Optional[int] = Field(None, ge=1, description="End page number (1-indexed)")
    prompt: str = Field(default="Parse these pages", description="Processing prompt")
    window_size: int = Field(default=2, ge=1, le=10, description="Window size")
    overlap: int = Field(default=1, ge=0, le=5, description="Window overlap")
    output_format: str = Field(default="json", description="Output format")


class DocumentUpload(BaseModel):
    """Document upload model"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    content: str = Field(..., description="Base64 encoded file content")
    
    @property
    def decoded_content(self) -> bytes:
        """Decode base64 content to bytes"""
        return base64.b64decode(self.content)


class BatchProcessRequest(BaseModel):
    """Request model for batch processing multiple documents"""
    documents: List[DocumentUpload] = Field(..., description="List of documents to process")
    prompt: str = Field(default="Parse and analyze these documents", description="Processing prompt")
    window_size: int = Field(default=2, description="Window size")
    overlap: int = Field(default=1, description="Window overlap")
    output_format: str = Field(default="json", description="Output format")
    merge_results: bool = Field(default=False, description="Merge results from all documents")


class ConfigUpdateRequest(BaseModel):
    """Request model for updating service configuration"""
    window_size: Optional[int] = Field(None, ge=1, le=10)
    overlap: Optional[int] = Field(None, ge=0, le=5)
    semantic_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_batch_size: Optional[int] = Field(None, ge=1, le=32)
    enable_caching: Optional[bool] = Field(None)
    cache_ttl: Optional[int] = Field(None, ge=60, le=86400)  # 1 minute to 1 day


class HealthCheckRequest(BaseModel):
    """Request model for health check with optional details"""
    include_model_status: bool = Field(default=False, description="Include model loading status")
    include_gpu_status: bool = Field(default=False, description="Include GPU status")
    include_memory_status: bool = Field(default=False, description="Include memory usage")


class WebhookConfig(BaseModel):
    """Webhook configuration for job completion notifications"""
    url: str = Field(..., description="Webhook URL")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    events: List[str] = Field(default=["job.completed", "job.failed"], description="Events to notify")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional headers")


class JobRequest(BaseModel):
    """Base request model for async job processing"""
    webhook: Optional[WebhookConfig] = Field(None, description="Webhook configuration")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=highest, 10=lowest)")
    timeout: int = Field(default=3600, ge=60, le=7200, description="Job timeout in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Extended requests that inherit from JobRequest for async processing
class AsyncProcessPDFRequest(ProcessPDFRequest, JobRequest):
    """Async PDF processing request with job management"""
    pass


class AsyncBatchProcessRequest(BatchProcessRequest, JobRequest):
    """Async batch processing request with job management"""
    pass 
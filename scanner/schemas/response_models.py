"""
Response models for the Dolphin FastAPI service
Compatible with OpenAI API format
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FinishReason(str, Enum):
    """OpenAI-compatible finish reasons"""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class Usage(BaseModel):
    """OpenAI-compatible usage statistics"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatChoice(BaseModel):
    """OpenAI-compatible chat completion choice"""
    index: int = Field(..., description="Choice index")
    message: Dict[str, str] = Field(..., description="Message content")
    finish_reason: FinishReason = Field(..., description="Reason for completion")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatChoice] = Field(..., description="List of completion choices")
    usage: Usage = Field(..., description="Token usage statistics")
    
    # Dolphin-specific metadata
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")


class WindowResult(BaseModel):
    """Result from processing a single window"""
    window_id: int = Field(..., description="Window identifier")
    start_page: int = Field(..., description="Starting page number")
    end_page: int = Field(..., description="Ending page number")
    elements: List[Dict[str, Any]] = Field(..., description="Extracted elements")
    processing_time: float = Field(..., description="Processing time in seconds")
    element_count: int = Field(..., description="Number of elements extracted")


class SemanticRelation(BaseModel):
    """Semantic relationship between elements"""
    source_window: int = Field(..., description="Source window ID")
    target_window: int = Field(..., description="Target window ID")
    similarity_score: float = Field(..., description="Similarity score")
    relation_type: str = Field(..., description="Type of relationship")
    source_element_id: str = Field(..., description="Source element identifier")
    target_element_id: str = Field(..., description="Target element identifier")


class CrossPageElement(BaseModel):
    """Element that spans multiple pages"""
    element_id: str = Field(..., description="Element identifier")
    element_type: str = Field(..., description="Element type (table, formula, text)")
    start_page: int = Field(..., description="Starting page")
    end_page: int = Field(..., description="Ending page")
    merged_content: str = Field(..., description="Merged element content")
    confidence_score: float = Field(..., description="Merge confidence score")


class Paragraph(BaseModel):
    """Discrete paragraph information"""
    paragraph_id: str = Field(..., description="Paragraph identifier")
    content: str = Field(..., description="Paragraph content")
    page_number: int = Field(..., description="Page number")
    position: Dict[str, float] = Field(..., description="Position coordinates")
    reading_order: int = Field(..., description="Reading order index")
    element_type: str = Field(..., description="Element type")


class MergedContent(BaseModel):
    """Merged and processed content"""
    text_content: str = Field(..., description="Merged text content")
    structured_data: Dict[str, Any] = Field(..., description="Structured data extraction")
    tables: List[Dict[str, Any]] = Field(..., description="Extracted tables")
    formulas: List[Dict[str, Any]] = Field(..., description="Extracted formulas")
    images: List[Dict[str, Any]] = Field(..., description="Extracted images")
    metadata: Dict[str, Any] = Field(..., description="Content metadata")


class ProcessingResult(BaseModel):
    """Main processing result"""
    document_id: str = Field(..., description="Document identifier")
    total_pages: int = Field(..., description="Total number of pages")
    processing_windows: List[WindowResult] = Field(..., description="Window processing results")
    merged_content: MergedContent = Field(..., description="Merged content")
    semantic_relationships: List[SemanticRelation] = Field(..., description="Semantic relationships")
    discrete_paragraphs: List[Paragraph] = Field(..., description="Discrete paragraphs")
    cross_page_elements: List[CrossPageElement] = Field(..., description="Cross-page elements")
    
    # Processing metadata
    processing_time: float = Field(..., description="Total processing time in seconds")
    window_count: int = Field(..., description="Number of windows processed")
    overlap_detections: int = Field(..., description="Number of overlaps detected")
    merge_operations: int = Field(..., description="Number of merge operations performed")
    
    # Quality metrics
    confidence_scores: Dict[str, float] = Field(..., description="Processing confidence scores")
    error_count: int = Field(default=0, description="Number of processing errors")
    warnings: List[str] = Field(default=[], description="Processing warnings")
    
    @property
    def formatted_content(self) -> str:
        """Get formatted content for OpenAI compatibility"""
        return self.merged_content.text_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return self.dict()
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        markdown_content = f"# Document Analysis Results\n\n"
        markdown_content += f"**Document ID:** {self.document_id}\n"
        markdown_content += f"**Total Pages:** {self.total_pages}\n"
        markdown_content += f"**Processing Time:** {self.processing_time:.2f} seconds\n\n"
        
        # Add merged content
        markdown_content += "## Content\n\n"
        markdown_content += self.merged_content.text_content
        
        # Add tables if present
        if self.merged_content.tables:
            markdown_content += "\n\n## Tables\n\n"
            for i, table in enumerate(self.merged_content.tables, 1):
                markdown_content += f"### Table {i}\n\n"
                markdown_content += table.get('content', '') + "\n\n"
        
        # Add formulas if present
        if self.merged_content.formulas:
            markdown_content += "\n\n## Mathematical Formulas\n\n"
            for i, formula in enumerate(self.merged_content.formulas, 1):
                markdown_content += f"**Formula {i}:** {formula.get('content', '')}\n\n"
        
        return markdown_content
    
    def to_html(self) -> str:
        """Convert to HTML format"""
        html_content = f"""
        <html>
        <head>
            <title>Document Analysis - {self.document_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .formula {{ background-color: #f9f9f9; padding: 5px; margin: 5px 0; border-left: 3px solid #007cba; }}
            </style>
        </head>
        <body>
            <h1>Document Analysis Results</h1>
            <div class="metadata">
                <p><strong>Document ID:</strong> {self.document_id}</p>
                <p><strong>Total Pages:</strong> {self.total_pages}</p>
                <p><strong>Processing Time:</strong> {self.processing_time:.2f} seconds</p>
            </div>
            
            <div class="content">
                <h2>Content</h2>
                <p>{self.merged_content.text_content.replace('\n', '<br>')}</p>
            </div>
        """
        
        if self.merged_content.tables:
            html_content += "<h2>Tables</h2>"
            for i, table in enumerate(self.merged_content.tables, 1):
                html_content += f"<h3>Table {i}</h3>"
                html_content += f"<div class='table'>{table.get('content', '')}</div>"
        
        if self.merged_content.formulas:
            html_content += "<h2>Mathematical Formulas</h2>"
            for i, formula in enumerate(self.merged_content.formulas, 1):
                html_content += f"<div class='formula'><strong>Formula {i}:</strong> {formula.get('content', '')}</div>"
        
        html_content += "</body></html>"
        return html_content


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    result_url: Optional[str] = Field(None, description="URL to fetch results")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    model_status: Optional[str] = Field(None, description="Model loading status")
    gpu_available: Optional[bool] = Field(None, description="GPU availability")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")
    active_jobs: Optional[int] = Field(None, description="Number of active jobs")
    queue_size: Optional[int] = Field(None, description="Queue size")


class BatchProcessingResponse(BaseModel):
    """Batch processing response"""
    batch_id: str = Field(..., description="Batch identifier")
    total_documents: int = Field(..., description="Total number of documents")
    completed_documents: int = Field(..., description="Number of completed documents")
    failed_documents: int = Field(..., description="Number of failed documents")
    results: List[ProcessingResult] = Field(..., description="Individual processing results")
    merged_result: Optional[ProcessingResult] = Field(None, description="Merged result if requested")
    processing_time: float = Field(..., description="Total batch processing time")
    errors: List[ErrorResponse] = Field(default=[], description="Processing errors")


class ConfigurationResponse(BaseModel):
    """Configuration response"""
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    updated_fields: List[str] = Field(..., description="Fields that were updated")
    timestamp: datetime = Field(..., description="Update timestamp")


class MetricsResponse(BaseModel):
    """Service metrics response"""
    requests_total: int = Field(..., description="Total number of requests")
    requests_per_minute: float = Field(..., description="Requests per minute")
    average_processing_time: float = Field(..., description="Average processing time")
    active_connections: int = Field(..., description="Active connections")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage")
    gpu_utilization: Optional[float] = Field(None, description="GPU utilization percentage")
    queue_metrics: Dict[str, int] = Field(..., description="Queue metrics")
    error_rate: float = Field(..., description="Error rate percentage")


class ListJobsResponse(BaseModel):
    """List jobs response"""
    jobs: List[JobStatusResponse] = Field(..., description="List of jobs")
    total_count: int = Field(..., description="Total number of jobs")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Has next page")


class StreamResponse(BaseModel):
    """Streaming response chunk"""
    chunk_id: int = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(..., description="Is final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk metadata") 
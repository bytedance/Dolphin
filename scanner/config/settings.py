"""
Configuration settings for the Dolphin FastAPI service
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model Configuration
    model_config_path: str = Field(
        default="../config/Dolphin.yaml",
        description="Path to Dolphin model configuration file"
    )
    
    model_checkpoint_path: str = Field(
        default="../checkpoints",
        description="Path to model checkpoints directory"
    )
    
    # Processing Settings
    default_window_size: int = Field(
        default=2,
        description="Default number of pages per sliding window"
    )
    
    default_overlap: int = Field(
        default=1,
        description="Default overlap between windows"
    )
    
    default_semantic_threshold: float = Field(
        default=0.8,
        description="Default similarity threshold for semantic overlap detection"
    )
    
    max_batch_size: int = Field(
        default=4,
        description="Maximum batch size for model inference"
    )
    
    # File Processing
    max_file_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum file size in bytes"
    )
    
    allowed_file_types: list = Field(
        default=[".pdf"],
        description="Allowed file extensions"
    )
    
    temp_upload_dir: str = Field(
        default="./temp_uploads",
        description="Temporary directory for file uploads"
    )
    
    # Performance Settings
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    
    cache_ttl: int = Field(
        default=3600,  # 1 hour
        description="Cache time-to-live in seconds"
    )
    
    # Redis Configuration (for caching and job queues)
    redis_host: str = Field(
        default="localhost",
        description="Redis host for caching and job management"
    )
    
    redis_port: int = Field(
        default=6379,
        description="Redis port"
    )
    
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    
    redis_db: int = Field(
        default=0,
        description="Redis database number"
    )
    
    # API Configuration
    api_version: str = Field(
        default="v1",
        description="API version"
    )
    
    max_concurrent_jobs: int = Field(
        default=10,
        description="Maximum number of concurrent processing jobs"
    )
    
    job_timeout: int = Field(
        default=3600,  # 1 hour
        description="Job timeout in seconds"
    )
    
    # Security
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for console only)"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    
    metrics_port: int = Field(
        default=9090,
        description="Port for metrics endpoint"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOLPHIN_",
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def setup_directories():
    """Create necessary directories"""
    os.makedirs(settings.temp_upload_dir, exist_ok=True)
    
    # Create results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "recognition_json"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "markdown", "figures"), exist_ok=True)


if __name__ == "__main__":
    # Print current settings for debugging
    print("Current Dolphin Service Settings:")
    print(f"Model Config Path: {settings.model_config_path}")
    print(f"Default Window Size: {settings.default_window_size}")
    print(f"Default Overlap: {settings.default_overlap}")
    print(f"Max File Size: {settings.max_file_size / (1024*1024):.1f} MB")
    print(f"Redis Host: {settings.redis_host}:{settings.redis_port}")
    print(f"Log Level: {settings.log_level}") 
#!/usr/bin/env python3
"""
Startup script for Dolphin FastAPI Service
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, setup_directories


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def check_dependencies():
    """Check if all required dependencies are available"""
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'sentence_transformers',
        'pymupdf',
        'pillow',
        'omegaconf',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are available")
    return True


def check_model_files(settings: Settings):
    """Check if Dolphin model files are available"""
    
    config_path = settings.model_config_path
    checkpoint_path = settings.model_checkpoint_path
    
    if not os.path.exists(config_path):
        print(f"❌ Model config file not found: {config_path}")
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint directory not found: {checkpoint_path}")
        return False
    
    print("✅ Model files are available")
    return True


def main():
    parser = argparse.ArgumentParser(description="Start Dolphin FastAPI Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("🐬 Starting Dolphin FastAPI Service")
    
    # Load settings
    try:
        settings = Settings()
        logger.info(f"Loaded configuration from {settings.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        sys.exit(1)
    
    # Setup directories
    try:
        setup_directories()
        logger.info("Created necessary directories")
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files(settings):
        logger.warning("Model files not found. Service may not work properly.")
        if args.check_only:
            sys.exit(1)
    
    if args.check_only:
        print("✅ All checks passed!")
        sys.exit(0)
    
    # Import and start the FastAPI app
    try:
        import uvicorn
        from main import app
        
        logger.info(f"Starting server on {args.host}:{args.port}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Reload: {args.reload}")
        logger.info(f"Documentation available at: http://{args.host}:{args.port}/docs")
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,  # Reload only works with 1 worker
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
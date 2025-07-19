"""
Dolphin model wrapper for FastAPI integration
Provides async interface and caching capabilities
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import torch
from omegaconf import OmegaConf
from PIL import Image
import time

# Add parent directory to path to import original Dolphin modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from chat import DOLPHIN
from utils.utils import convert_pdf_to_images, prepare_image, process_elements
from utils.processor import DolphinProcessor

logger = logging.getLogger(__name__)


class DolphinModelWrapper:
    """
    Async wrapper for the Dolphin model with caching and batch processing
    """
    
    def __init__(self, config_path: str):
        """Initialize the Dolphin model wrapper
        
        Args:
            config_path: Path to the Dolphin configuration file
        """
        self.config_path = config_path
        self.model = None
        self.config = None
        self.device = None
        self.is_loaded = False
        self.loading_lock = asyncio.Lock()
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Cache for repeated requests (optional)
        self.cache = {}
        self.cache_enabled = True
        self.max_cache_size = 100
        
        # Load model synchronously during initialization
        self._load_model()
    
    def _load_model(self):
        """Load the Dolphin model synchronously"""
        try:
            logger.info(f"Loading Dolphin model from config: {self.config_path}")
            
            # Load configuration
            self.config = OmegaConf.load(self.config_path)
            
            # Initialize Dolphin model
            self.model = DOLPHIN(self.config)
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Model loaded successfully on device: {self.device}")
            
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load Dolphin model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    async def ensure_loaded(self):
        """Ensure the model is loaded (async-safe)"""
        if not self.is_loaded:
            async with self.loading_lock:
                if not self.is_loaded:
                    await asyncio.get_event_loop().run_in_executor(None, self._load_model)
    
    def _get_cache_key(self, prompt: str, image_hash: str) -> str:
        """Generate cache key for request"""
        return f"{hash(prompt)}_{image_hash}"
    
    def _hash_image(self, image: Image.Image) -> str:
        """Generate hash for PIL Image"""
        import hashlib
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()[:16]
    
    async def chat_async(
        self,
        prompt: str,
        image: Union[Image.Image, List[Image.Image]],
        use_cache: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Async wrapper for the chat method
        
        Args:
            prompt: Text prompt for the model
            image: PIL Image or list of images
            use_cache: Whether to use caching
            **kwargs: Additional arguments for the model
            
        Returns:
            Generated text or list of texts
        """
        await self.ensure_loaded()
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Handle caching for single images
            if use_cache and self.cache_enabled and isinstance(image, Image.Image):
                cache_key = self._get_cache_key(prompt, self._hash_image(image))
                if cache_key in self.cache:
                    logger.debug("Cache hit for request")
                    return self.cache[cache_key]
            
            # Run model inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.model.chat(prompt, image, **kwargs)
            )
            
            # Cache result if applicable
            if (use_cache and self.cache_enabled and isinstance(image, Image.Image) and 
                len(self.cache) < self.max_cache_size):
                cache_key = self._get_cache_key(prompt, self._hash_image(image))
                self.cache[cache_key] = result
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.debug(f"Model inference completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Model inference failed: {str(e)}")
            raise
    
    async def process_pdf_images(
        self,
        pdf_images: List[Image.Image],
        prompt: str = "Parse the reading order of this document.",
        max_batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process a list of PDF page images
        
        Args:
            pdf_images: List of PIL Images from PDF pages
            prompt: Prompt for processing
            max_batch_size: Maximum batch size for processing
            
        Returns:
            List of processing results for each page
        """
        await self.ensure_loaded()
        
        results = []
        
        for page_idx, pil_image in enumerate(pdf_images):
            logger.info(f"Processing page {page_idx + 1}/{len(pdf_images)}")
            
            try:
                # Stage 1: Page-level layout and reading order parsing
                layout_output = await self.chat_async(prompt, pil_image)
                
                # Stage 2: Element-level content parsing
                padded_image, dims = prepare_image(pil_image)
                
                # Process elements (run in executor to avoid blocking)
                loop = asyncio.get_event_loop()
                recognition_results = await loop.run_in_executor(
                    None,
                    lambda: process_elements(
                        layout_output, padded_image, dims, self.model, 
                        max_batch_size, None, f"page_{page_idx + 1:03d}"
                    )
                )
                
                # Create page result
                page_result = {
                    "page_number": page_idx + 1,
                    "layout_output": layout_output,
                    "elements": recognition_results,
                    "processing_time": time.time()
                }
                
                results.append(page_result)
                
            except Exception as e:
                logger.error(f"Error processing page {page_idx + 1}: {str(e)}")
                # Add error result
                results.append({
                    "page_number": page_idx + 1,
                    "error": str(e),
                    "elements": []
                })
        
        return results
    
    async def convert_pdf_to_images_async(self, pdf_content: bytes, target_size: int = 896) -> List[Image.Image]:
        """
        Convert PDF bytes to images asynchronously
        
        Args:
            pdf_content: PDF file content as bytes
            target_size: Target size for the longest dimension
            
        Returns:
            List of PIL Images
        """
        import io
        import tempfile
        
        # Write PDF content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name
        
        try:
            # Run PDF conversion in executor
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(
                None,
                lambda: convert_pdf_to_images(temp_path, target_size)
            )
            return images
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    async def process_single_element(
        self,
        image: Image.Image,
        element_type: str = "text",
        prompt: Optional[str] = None
    ) -> str:
        """
        Process a single document element
        
        Args:
            image: PIL Image of the element
            element_type: Type of element (text, table, formula)
            prompt: Custom prompt (optional)
            
        Returns:
            Extracted content as string
        """
        await self.ensure_loaded()
        
        # Select appropriate prompt based on element type
        if prompt is None:
            if element_type == "table":
                prompt = "Parse the table in the image."
            elif element_type == "formula":
                prompt = "Read text in the image."
            else:  # Default to text
                prompt = "Read text in the image."
        
        # Process the element
        result = await self.chat_async(prompt, image)
        return result.strip() if result else ""
    
    async def batch_process_elements(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_batch_size: int = 8
    ) -> List[str]:
        """
        Process multiple elements in batches
        
        Args:
            images: List of PIL Images
            prompts: List of prompts (same length as images)
            max_batch_size: Maximum batch size
            
        Returns:
            List of processing results
        """
        await self.ensure_loaded()
        
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), max_batch_size):
            batch_images = images[i:i + max_batch_size]
            batch_prompts = prompts[i:i + max_batch_size]
            
            try:
                # Process batch
                batch_results = await self.chat_async(batch_prompts, batch_images)
                
                # Ensure batch_results is a list
                if not isinstance(batch_results, list):
                    batch_results = [batch_results]
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//max_batch_size + 1}: {str(e)}")
                # Add error results for the batch
                error_results = [f"Error: {str(e)}"] * len(batch_images)
                results.extend(error_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "cache_size": len(self.cache),
            "cache_enabled": self.cache_enabled
        }
    
    def clear_cache(self):
        """Clear the request cache"""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info("Model statistics reset")


# Global model instance (singleton pattern)
_global_model_instance: Optional[DolphinModelWrapper] = None


def get_dolphin_model(config_path: str = None) -> DolphinModelWrapper:
    """
    Get or create global Dolphin model instance
    
    Args:
        config_path: Path to configuration file (only used for first initialization)
        
    Returns:
        DolphinModelWrapper instance
    """
    global _global_model_instance
    
    if _global_model_instance is None:
        if config_path is None:
            raise ValueError("config_path required for first initialization")
        _global_model_instance = DolphinModelWrapper(config_path)
    
    return _global_model_instance 
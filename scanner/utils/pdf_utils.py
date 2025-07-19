"""
Utility functions for PDF handling and processing
"""

import io
import logging
import mimetypes
import tempfile
import os
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
import pymupdf
import hashlib

logger = logging.getLogger(__name__)


def validate_pdf_file(file_content: bytes) -> Tuple[bool, str]:
    """
    Validate if the file content is a valid PDF
    
    Args:
        file_content: File content as bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check file size
        if len(file_content) == 0:
            return False, "Empty file"
        
        if len(file_content) > 100 * 1024 * 1024:  # 100MB limit
            return False, "File too large (max 100MB)"
        
        # Check PDF header
        if not file_content.startswith(b'%PDF'):
            return False, "Not a valid PDF file"
        
        # Try to open with PyMuPDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            doc = pymupdf.open(temp_path)
            page_count = len(doc)
            doc.close()
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            if page_count > 500:  # Reasonable limit
                return False, f"PDF has too many pages ({page_count}). Maximum allowed: 500"
            
            return True, ""
            
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        return False, f"Error validating PDF: {str(e)}"


def get_pdf_metadata(file_content: bytes) -> Dict[str, Any]:
    """
    Extract metadata from PDF file
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        Dictionary with PDF metadata
    """
    metadata = {
        'page_count': 0,
        'file_size': len(file_content),
        'title': '',
        'author': '',
        'subject': '',
        'creator': '',
        'producer': '',
        'creation_date': None,
        'modification_date': None,
        'encrypted': False,
        'permissions': {},
        'page_dimensions': []
    }
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            doc = pymupdf.open(temp_path)
            
            # Basic info
            metadata['page_count'] = len(doc)
            metadata['encrypted'] = doc.needs_pass
            
            # Document metadata
            doc_metadata = doc.metadata
            metadata['title'] = doc_metadata.get('title', '')
            metadata['author'] = doc_metadata.get('author', '')
            metadata['subject'] = doc_metadata.get('subject', '')
            metadata['creator'] = doc_metadata.get('creator', '')
            metadata['producer'] = doc_metadata.get('producer', '')
            metadata['creation_date'] = doc_metadata.get('creationDate', '')
            metadata['modification_date'] = doc_metadata.get('modDate', '')
            
            # Page dimensions
            for page_num in range(min(len(doc), 10)):  # Check first 10 pages
                page = doc[page_num]
                rect = page.rect
                metadata['page_dimensions'].append({
                    'page': page_num + 1,
                    'width': rect.width,
                    'height': rect.height,
                    'orientation': 'portrait' if rect.height > rect.width else 'landscape'
                })
            
            # Permissions (if not encrypted)
            if not doc.needs_pass:
                metadata['permissions'] = {
                    'print': doc.permissions & pymupdf.PDF_PERM_PRINT,
                    'modify': doc.permissions & pymupdf.PDF_PERM_MODIFY,
                    'copy': doc.permissions & pymupdf.PDF_PERM_COPY,
                    'annotate': doc.permissions & pymupdf.PDF_PERM_ANNOTATE
                }
            
            doc.close()
            
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {str(e)}")
        metadata['error'] = str(e)
    
    return metadata


def optimize_pdf_for_processing(file_content: bytes) -> bytes:
    """
    Optimize PDF for processing (remove unnecessary elements, compress)
    
    Args:
        file_content: Original PDF content
        
    Returns:
        Optimized PDF content
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_input:
            temp_input.write(file_content)
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            # Open and process PDF
            doc = pymupdf.open(input_path)
            
            # Create new document
            new_doc = pymupdf.open()
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Remove annotations and links for faster processing
                page.delete_annot_list(page.annots())
                page.delete_link_list(page.get_links())
                
                # Insert page into new document
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Save optimized document
            new_doc.save(output_path, garbage=4, deflate=True, clean=True)
            
            doc.close()
            new_doc.close()
            
            # Read optimized content
            with open(output_path, 'rb') as f:
                optimized_content = f.read()
            
            logger.info(f"PDF optimized: {len(file_content)} -> {len(optimized_content)} bytes")
            return optimized_content
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    except Exception as e:
        logger.warning(f"PDF optimization failed, using original: {str(e)}")
        return file_content


def split_pdf_by_pages(file_content: bytes, pages_per_chunk: int = 10) -> List[bytes]:
    """
    Split a large PDF into smaller chunks
    
    Args:
        file_content: PDF file content
        pages_per_chunk: Number of pages per chunk
        
    Returns:
        List of PDF chunks as bytes
    """
    chunks = []
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            doc = pymupdf.open(temp_path)
            total_pages = len(doc)
            
            for start_page in range(0, total_pages, pages_per_chunk):
                end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)
                
                # Create chunk document
                chunk_doc = pymupdf.open()
                chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)
                
                # Save chunk to bytes
                with tempfile.NamedTemporaryFile(suffix='.pdf') as chunk_temp:
                    chunk_doc.save(chunk_temp.name)
                    chunk_temp.seek(0)
                    chunk_content = chunk_temp.read()
                    chunks.append(chunk_content)
                
                chunk_doc.close()
                
                logger.info(f"Created chunk {len(chunks)}: pages {start_page + 1}-{end_page + 1}")
            
            doc.close()
            
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error splitting PDF: {str(e)}")
        # Return original as single chunk if splitting fails
        chunks = [file_content]
    
    return chunks


def extract_text_from_pdf(file_content: bytes) -> Dict[int, str]:
    """
    Extract text from PDF pages using PyMuPDF
    
    Args:
        file_content: PDF file content
        
    Returns:
        Dictionary mapping page numbers to extracted text
    """
    page_texts = {}
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            doc = pymupdf.open(temp_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                page_texts[page_num + 1] = text  # 1-indexed
            
            doc.close()
            
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
    
    return page_texts


def convert_pdf_pages_to_images(
    file_content: bytes,
    page_numbers: Optional[List[int]] = None,
    target_size: int = 896,
    image_format: str = "RGB"
) -> List[Image.Image]:
    """
    Convert specific PDF pages to images
    
    Args:
        file_content: PDF file content
        page_numbers: List of page numbers to convert (1-indexed, None for all)
        target_size: Target size for the longest dimension
        image_format: PIL image format
        
    Returns:
        List of PIL Images
    """
    images = []
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            doc = pymupdf.open(temp_path)
            
            # Determine which pages to process
            if page_numbers is None:
                pages_to_process = range(len(doc))
            else:
                pages_to_process = [p - 1 for p in page_numbers if 1 <= p <= len(doc)]  # Convert to 0-indexed
            
            for page_idx in pages_to_process:
                page = doc[page_idx]
                
                # Calculate scale to make longest dimension equal to target_size
                rect = page.rect
                scale = target_size / max(rect.width, rect.height)
                
                # Render page as image
                mat = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to specified format
                if image_format != pil_image.mode:
                    pil_image = pil_image.convert(image_format)
                
                images.append(pil_image)
            
            doc.close()
            
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error converting PDF pages to images: {str(e)}")
    
    return images


def get_pdf_hash(file_content: bytes) -> str:
    """
    Generate a hash for PDF content (for caching and deduplication)
    
    Args:
        file_content: PDF file content
        
    Returns:
        SHA-256 hash of the content
    """
    return hashlib.sha256(file_content).hexdigest()


def detect_pdf_language(file_content: bytes) -> str:
    """
    Detect the primary language of the PDF content
    
    Args:
        file_content: PDF file content
        
    Returns:
        Detected language code (e.g., 'en', 'zh', 'ja')
    """
    try:
        # Extract text from first few pages
        page_texts = extract_text_from_pdf(file_content)
        
        # Combine text from first 3 pages
        combined_text = ""
        for page_num in sorted(page_texts.keys())[:3]:
            combined_text += page_texts[page_num] + " "
        
        if not combined_text.strip():
            return "unknown"
        
        # Simple language detection based on character patterns
        # This is a basic implementation - for production, use langdetect or similar
        
        # Check for Chinese characters
        chinese_chars = sum(1 for char in combined_text if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(combined_text) * 0.1:
            return "zh"
        
        # Check for Japanese characters
        japanese_chars = sum(1 for char in combined_text if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff')
        if japanese_chars > len(combined_text) * 0.05:
            return "ja"
        
        # Default to English for Latin scripts
        return "en"
    
    except Exception as e:
        logger.error(f"Error detecting PDF language: {str(e)}")
        return "unknown"


def estimate_processing_time(file_content: bytes) -> float:
    """
    Estimate processing time based on PDF characteristics
    
    Args:
        file_content: PDF file content
        
    Returns:
        Estimated processing time in seconds
    """
    try:
        metadata = get_pdf_metadata(file_content)
        page_count = metadata.get('page_count', 1)
        file_size_mb = metadata.get('file_size', 0) / (1024 * 1024)
        
        # Base time per page (seconds)
        base_time_per_page = 2.0
        
        # Adjust for file size
        size_factor = min(file_size_mb / 10.0, 2.0)  # Cap at 2x for very large files
        
        # Adjust for page count (efficiency improves with batch processing)
        if page_count > 10:
            efficiency_factor = 0.8
        else:
            efficiency_factor = 1.0
        
        estimated_time = page_count * base_time_per_page * (1 + size_factor) * efficiency_factor
        
        return max(estimated_time, 5.0)  # Minimum 5 seconds
    
    except Exception:
        return 30.0  # Default estimate 
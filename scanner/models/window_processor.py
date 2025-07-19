"""
Sliding window processor for PDF page management
Handles overlapping windows and page combinations
"""

import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import cv2
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Window:
    """Represents a sliding window of pages"""
    window_id: int
    start_page: int
    end_page: int
    pages: List[Image.Image]
    page_numbers: List[int]
    
    @property
    def size(self) -> int:
        """Get window size (number of pages)"""
        return len(self.pages)
    
    @property
    def page_range(self) -> str:
        """Get human-readable page range"""
        if self.start_page == self.end_page:
            return f"Page {self.start_page}"
        return f"Pages {self.start_page}-{self.end_page}"


@dataclass
class WindowResult:
    """Result from processing a window"""
    window: Window
    layout_output: str
    elements: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error: Optional[str] = None
    
    @property
    def element_count(self) -> int:
        """Get number of elements in this window"""
        return len(self.elements)


class SlidingWindowProcessor:
    """
    Manages sliding windows for PDF page processing
    Handles overlapping windows and page combinations
    """
    
    def __init__(self, window_size: int = 2, overlap: int = 1):
        """
        Initialize sliding window processor
        
        Args:
            window_size: Number of pages per window
            overlap: Number of pages to overlap between windows
        """
        self.window_size = window_size
        self.overlap = overlap
        self.windows: List[Window] = []
        
        # Validation
        if window_size < 1:
            raise ValueError("Window size must be at least 1")
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")
        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")
        
        logger.info(f"Initialized SlidingWindowProcessor: size={window_size}, overlap={overlap}")
    
    def create_windows(self, pages: List[Image.Image]) -> List[Window]:
        """
        Create sliding windows from a list of PDF pages
        
        Args:
            pages: List of PIL Images representing PDF pages
            
        Returns:
            List of Window objects
        """
        if not pages:
            return []
        
        total_pages = len(pages)
        windows = []
        window_id = 0
        
        # Calculate step size (how many pages to advance each window)
        step_size = self.window_size - self.overlap
        
        # Create windows
        for start_idx in range(0, total_pages, step_size):
            end_idx = min(start_idx + self.window_size, total_pages)
            
            # Skip if we don't have enough pages for a meaningful window
            if end_idx - start_idx < 1:
                break
            
            window_pages = pages[start_idx:end_idx]
            page_numbers = list(range(start_idx + 1, end_idx + 1))  # 1-indexed
            
            window = Window(
                window_id=window_id,
                start_page=start_idx + 1,  # 1-indexed
                end_page=end_idx,  # 1-indexed
                pages=window_pages,
                page_numbers=page_numbers
            )
            
            windows.append(window)
            window_id += 1
            
            logger.debug(f"Created window {window_id}: {window.page_range}")
        
        self.windows = windows
        logger.info(f"Created {len(windows)} windows from {total_pages} pages")
        
        return windows
    
    def combine_window_pages(self, window: Window, method: str = "vertical") -> Image.Image:
        """
        Combine multiple pages in a window into a single image
        
        Args:
            window: Window containing pages to combine
            method: Combination method ("vertical", "horizontal", "grid")
            
        Returns:
            Combined PIL Image
        """
        if not window.pages:
            raise ValueError("Window has no pages to combine")
        
        if len(window.pages) == 1:
            return window.pages[0]
        
        try:
            if method == "vertical":
                return self._combine_vertical(window.pages)
            elif method == "horizontal":
                return self._combine_horizontal(window.pages)
            elif method == "grid":
                return self._combine_grid(window.pages)
            else:
                raise ValueError(f"Unknown combination method: {method}")
                
        except Exception as e:
            logger.error(f"Error combining window pages: {str(e)}")
            # Fallback: return the first page
            return window.pages[0]
    
    def _combine_vertical(self, pages: List[Image.Image]) -> Image.Image:
        """Combine pages vertically (stacked)"""
        # Resize all pages to same width
        widths = [page.width for page in pages]
        target_width = max(widths)
        
        resized_pages = []
        total_height = 0
        
        for page in pages:
            if page.width != target_width:
                # Maintain aspect ratio
                ratio = target_width / page.width
                new_height = int(page.height * ratio)
                page = page.resize((target_width, new_height), Image.Resampling.LANCZOS)
            
            resized_pages.append(page)
            total_height += page.height
        
        # Create combined image
        combined = Image.new('RGB', (target_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for page in resized_pages:
            combined.paste(page, (0, y_offset))
            y_offset += page.height
        
        return combined
    
    def _combine_horizontal(self, pages: List[Image.Image]) -> Image.Image:
        """Combine pages horizontally (side by side)"""
        # Resize all pages to same height
        heights = [page.height for page in pages]
        target_height = max(heights)
        
        resized_pages = []
        total_width = 0
        
        for page in pages:
            if page.height != target_height:
                # Maintain aspect ratio
                ratio = target_height / page.height
                new_width = int(page.width * ratio)
                page = page.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            resized_pages.append(page)
            total_width += page.width
        
        # Create combined image
        combined = Image.new('RGB', (total_width, target_height), (255, 255, 255))
        
        x_offset = 0
        for page in resized_pages:
            combined.paste(page, (x_offset, 0))
            x_offset += page.width
        
        return combined
    
    def _combine_grid(self, pages: List[Image.Image]) -> Image.Image:
        """Combine pages in a grid layout"""
        num_pages = len(pages)
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_pages)))
        rows = int(np.ceil(num_pages / cols))
        
        # Find common dimensions
        max_width = max(page.width for page in pages)
        max_height = max(page.height for page in pages)
        
        # Resize all pages to common size
        resized_pages = []
        for page in pages:
            if page.size != (max_width, max_height):
                page = page.resize((max_width, max_height), Image.Resampling.LANCZOS)
            resized_pages.append(page)
        
        # Create grid
        grid_width = cols * max_width
        grid_height = rows * max_height
        combined = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for i, page in enumerate(resized_pages):
            row = i // cols
            col = i % cols
            x = col * max_width
            y = row * max_height
            combined.paste(page, (x, y))
        
        return combined
    
    async def process_windows_async(
        self,
        windows: List[Window],
        dolphin_model,
        prompt: str = "Parse the reading order of this document.",
        combination_method: str = "vertical",
        max_batch_size: int = 4
    ) -> List[WindowResult]:
        """
        Process all windows asynchronously
        
        Args:
            windows: List of windows to process
            dolphin_model: DolphinModelWrapper instance
            prompt: Processing prompt
            combination_method: How to combine pages in each window
            max_batch_size: Maximum batch size for processing
            
        Returns:
            List of WindowResult objects
        """
        results = []
        
        for window in windows:
            logger.info(f"Processing {window.page_range} (Window {window.window_id})")
            start_time = time.time()
            
            try:
                # Combine pages in the window
                combined_image = self.combine_window_pages(window, combination_method)
                
                # Process the combined image
                layout_output = await dolphin_model.chat_async(prompt, combined_image)
                
                # Process individual pages for element extraction
                all_elements = []
                for page_idx, page in enumerate(window.pages):
                    page_results = await dolphin_model.process_pdf_images(
                        [page], prompt, max_batch_size
                    )
                    
                    if page_results and page_results[0].get('elements'):
                        # Add window and page metadata to elements
                        for element in page_results[0]['elements']:
                            element['window_id'] = window.window_id
                            element['page_number'] = window.page_numbers[page_idx]
                            element['window_page_index'] = page_idx
                        
                        all_elements.extend(page_results[0]['elements'])
                
                processing_time = time.time() - start_time
                
                result = WindowResult(
                    window=window,
                    layout_output=layout_output,
                    elements=all_elements,
                    processing_time=processing_time,
                    success=True
                )
                
                logger.info(f"Successfully processed {window.page_range} in {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Error processing {window.page_range}: {str(e)}")
                
                result = WindowResult(
                    window=window,
                    layout_output="",
                    elements=[],
                    processing_time=processing_time,
                    success=False,
                    error=str(e)
                )
            
            results.append(result)
        
        return results
    
    def get_overlap_info(self) -> List[Dict[str, Any]]:
        """
        Get information about overlapping regions between windows
        
        Returns:
            List of overlap information dictionaries
        """
        overlaps = []
        
        for i in range(len(self.windows) - 1):
            current_window = self.windows[i]
            next_window = self.windows[i + 1]
            
            # Find overlapping pages
            current_pages = set(current_window.page_numbers)
            next_pages = set(next_window.page_numbers)
            overlap_pages = current_pages.intersection(next_pages)
            
            if overlap_pages:
                overlaps.append({
                    'window1_id': current_window.window_id,
                    'window2_id': next_window.window_id,
                    'window1_range': current_window.page_range,
                    'window2_range': next_window.page_range,
                    'overlap_pages': sorted(list(overlap_pages)),
                    'overlap_count': len(overlap_pages)
                })
        
        return overlaps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        if not self.windows:
            return {
                'total_windows': 0,
                'total_pages': 0,
                'window_size': self.window_size,
                'overlap': self.overlap
            }
        
        total_pages = max(window.end_page for window in self.windows)
        overlap_info = self.get_overlap_info()
        
        return {
            'total_windows': len(self.windows),
            'total_pages': total_pages,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.window_size - self.overlap,
            'overlap_regions': len(overlap_info),
            'average_elements_per_window': None,  # Will be calculated after processing
            'windows_info': [
                {
                    'window_id': w.window_id,
                    'page_range': w.page_range,
                    'size': w.size
                }
                for w in self.windows
            ]
        }
    
    def validate_windows(self) -> List[str]:
        """
        Validate window configuration and return any warnings
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        if not self.windows:
            warnings.append("No windows created")
            return warnings
        
        # Check for gaps
        all_pages = set()
        for window in self.windows:
            all_pages.update(window.page_numbers)
        
        total_pages = max(all_pages) if all_pages else 0
        expected_pages = set(range(1, total_pages + 1))
        missing_pages = expected_pages - all_pages
        
        if missing_pages:
            warnings.append(f"Missing pages in windows: {sorted(missing_pages)}")
        
        # Check overlap consistency
        if self.overlap > 0:
            overlap_info = self.get_overlap_info()
            for overlap in overlap_info:
                if overlap['overlap_count'] != self.overlap:
                    warnings.append(
                        f"Inconsistent overlap between windows {overlap['window1_id']} "
                        f"and {overlap['window2_id']}: expected {self.overlap}, "
                        f"got {overlap['overlap_count']}"
                    )
        
        return warnings 
"""
Unit tests for sliding window processor
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image

from models.window_processor import (
    SlidingWindowProcessor, Window, WindowResult
)


@pytest.fixture
def sample_images():
    """Create sample PIL images for testing"""
    images = []
    for i in range(6):  # Create 6 test images
        # Create a simple RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        images.append(img)
    return images


@pytest.fixture
def processor():
    """Create a SlidingWindowProcessor instance"""
    return SlidingWindowProcessor(window_size=2, overlap=1)


@pytest.fixture
def mock_dolphin_model():
    """Create a mock Dolphin model"""
    model = Mock()
    model.chat_async = AsyncMock(return_value="Mock layout output")
    model.process_pdf_images = AsyncMock(return_value=[
        {"page_number": 1, "elements": [{"label": "text", "text": "Sample text"}]}
    ])
    return model


class TestWindow:
    """Test Window dataclass"""
    
    def test_window_creation(self, sample_images):
        """Test Window object creation"""
        window_pages = sample_images[:2]
        window = Window(
            window_id=0,
            start_page=1,
            end_page=2,
            pages=window_pages,
            page_numbers=[1, 2]
        )
        
        assert window.window_id == 0
        assert window.start_page == 1
        assert window.end_page == 2
        assert len(window.pages) == 2
        assert window.page_numbers == [1, 2]
        assert window.size == 2
        assert window.page_range == "Pages 1-2"
    
    def test_single_page_window(self, sample_images):
        """Test Window with single page"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=1,
            pages=[sample_images[0]],
            page_numbers=[1]
        )
        
        assert window.size == 1
        assert window.page_range == "Page 1"
    
    def test_window_properties(self, sample_images):
        """Test Window properties"""
        window = Window(
            window_id=2,
            start_page=5,
            end_page=7,
            pages=sample_images[:3],
            page_numbers=[5, 6, 7]
        )
        
        assert window.size == 3
        assert window.page_range == "Pages 5-7"


class TestSlidingWindowProcessor:
    """Test SlidingWindowProcessor class"""
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = SlidingWindowProcessor(window_size=3, overlap=2)
        
        assert processor.window_size == 3
        assert processor.overlap == 2
        assert processor.windows == []
    
    def test_invalid_initialization(self):
        """Test invalid processor initialization"""
        with pytest.raises(ValueError, match="Window size must be at least 1"):
            SlidingWindowProcessor(window_size=0, overlap=1)
        
        with pytest.raises(ValueError, match="Overlap cannot be negative"):
            SlidingWindowProcessor(window_size=2, overlap=-1)
        
        with pytest.raises(ValueError, match="Overlap must be less than window size"):
            SlidingWindowProcessor(window_size=2, overlap=2)
    
    def test_create_windows_basic(self, processor, sample_images):
        """Test basic window creation"""
        windows = processor.create_windows(sample_images)
        
        # With 6 pages, window_size=2, overlap=1, we should get:
        # Window 0: pages 0-1 (pages 1-2)
        # Window 1: pages 1-2 (pages 2-3)  
        # Window 2: pages 2-3 (pages 3-4)
        # Window 3: pages 3-4 (pages 4-5)
        # Window 4: pages 4-5 (pages 5-6)
        
        assert len(windows) == 5
        
        # Check first window
        assert windows[0].window_id == 0
        assert windows[0].start_page == 1
        assert windows[0].end_page == 2
        assert windows[0].page_numbers == [1, 2]
        
        # Check last window
        assert windows[4].window_id == 4
        assert windows[4].start_page == 5
        assert windows[4].end_page == 6
        assert windows[4].page_numbers == [5, 6]
    
    def test_create_windows_no_overlap(self, sample_images):
        """Test window creation with no overlap"""
        processor = SlidingWindowProcessor(window_size=2, overlap=0)
        windows = processor.create_windows(sample_images)
        
        # With 6 pages, window_size=2, overlap=0:
        # Window 0: pages 1-2
        # Window 1: pages 3-4
        # Window 2: pages 5-6
        
        assert len(windows) == 3
        assert windows[0].page_numbers == [1, 2]
        assert windows[1].page_numbers == [3, 4]
        assert windows[2].page_numbers == [5, 6]
    
    def test_create_windows_large_window_size(self, sample_images):
        """Test window creation with large window size"""
        processor = SlidingWindowProcessor(window_size=4, overlap=1)
        windows = processor.create_windows(sample_images)
        
        # With 6 pages, window_size=4, overlap=1:
        # Window 0: pages 1-4
        # Window 1: pages 4-6 (only 3 pages)
        
        assert len(windows) == 2
        assert windows[0].page_numbers == [1, 2, 3, 4]
        assert windows[1].page_numbers == [4, 5, 6]
    
    def test_create_windows_empty_pages(self, processor):
        """Test window creation with empty page list"""
        windows = processor.create_windows([])
        assert len(windows) == 0
    
    def test_create_windows_single_page(self, processor, sample_images):
        """Test window creation with single page"""
        windows = processor.create_windows([sample_images[0]])
        
        assert len(windows) == 1
        assert windows[0].page_numbers == [1]
        assert windows[0].size == 1


class TestWindowCombination:
    """Test window page combination methods"""
    
    def test_combine_single_page(self, processor, sample_images):
        """Test combining window with single page"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=1,
            pages=[sample_images[0]],
            page_numbers=[1]
        )
        
        combined = processor.combine_window_pages(window)
        assert combined == sample_images[0]
    
    def test_combine_vertical(self, processor, sample_images):
        """Test vertical page combination"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=2,
            pages=sample_images[:2],
            page_numbers=[1, 2]
        )
        
        combined = processor.combine_window_pages(window, method="vertical")
        
        # Combined image should be taller than individual images
        assert isinstance(combined, Image.Image)
        assert combined.height > sample_images[0].height
        assert combined.width >= max(img.width for img in sample_images[:2])
    
    def test_combine_horizontal(self, processor, sample_images):
        """Test horizontal page combination"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=2,
            pages=sample_images[:2],
            page_numbers=[1, 2]
        )
        
        combined = processor.combine_window_pages(window, method="horizontal")
        
        # Combined image should be wider than individual images
        assert isinstance(combined, Image.Image)
        assert combined.width > sample_images[0].width
        assert combined.height >= max(img.height for img in sample_images[:2])
    
    def test_combine_grid(self, processor, sample_images):
        """Test grid page combination"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=4,
            pages=sample_images[:4],
            page_numbers=[1, 2, 3, 4]
        )
        
        combined = processor.combine_window_pages(window, method="grid")
        
        # Combined image should accommodate all pages in grid
        assert isinstance(combined, Image.Image)
        assert combined.width >= sample_images[0].width * 2  # At least 2x2 grid
        assert combined.height >= sample_images[0].height * 2
    
    def test_combine_invalid_method(self, processor, sample_images):
        """Test combining with invalid method"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=2,
            pages=sample_images[:2],
            page_numbers=[1, 2]
        )
        
        with pytest.raises(ValueError):
            processor.combine_window_pages(window, method="invalid")
    
    def test_combine_empty_window(self, processor):
        """Test combining empty window"""
        window = Window(
            window_id=0,
            start_page=1,
            end_page=1,
            pages=[],
            page_numbers=[]
        )
        
        with pytest.raises(ValueError):
            processor.combine_window_pages(window)


class TestAsyncProcessing:
    """Test async window processing"""
    
    @pytest.mark.asyncio
    async def test_process_windows_async(self, processor, sample_images, mock_dolphin_model):
        """Test async window processing"""
        windows = processor.create_windows(sample_images[:4])  # Use 4 images
        
        results = await processor.process_windows_async(
            windows=windows,
            dolphin_model=mock_dolphin_model,
            prompt="Test prompt"
        )
        
        assert len(results) == len(windows)
        
        for result in results:
            assert isinstance(result, WindowResult)
            assert result.success == True
            assert result.layout_output == "Mock layout output"
            assert result.processing_time > 0
            assert isinstance(result.elements, list)
    
    @pytest.mark.asyncio
    async def test_process_windows_with_error(self, processor, sample_images, mock_dolphin_model):
        """Test async window processing with errors"""
        # Make the model raise an exception
        mock_dolphin_model.chat_async.side_effect = Exception("Model error")
        
        windows = processor.create_windows(sample_images[:2])
        
        results = await processor.process_windows_async(
            windows=windows,
            dolphin_model=mock_dolphin_model,
            prompt="Test prompt"
        )
        
        assert len(results) == len(windows)
        
        for result in results:
            assert isinstance(result, WindowResult)
            assert result.success == False
            assert result.error is not None
            assert "Model error" in result.error
    
    @pytest.mark.asyncio
    async def test_process_empty_windows(self, processor, mock_dolphin_model):
        """Test processing empty window list"""
        results = await processor.process_windows_async(
            windows=[],
            dolphin_model=mock_dolphin_model,
            prompt="Test prompt"
        )
        
        assert len(results) == 0


class TestWindowAnalysis:
    """Test window analysis methods"""
    
    def test_get_overlap_info(self, processor, sample_images):
        """Test overlap information generation"""
        windows = processor.create_windows(sample_images)
        overlap_info = processor.get_overlap_info()
        
        assert len(overlap_info) == len(windows) - 1  # n-1 overlaps for n windows
        
        for i, overlap in enumerate(overlap_info):
            assert overlap['window1_id'] == i
            assert overlap['window2_id'] == i + 1
            assert overlap['overlap_count'] == processor.overlap
            assert len(overlap['overlap_pages']) == processor.overlap
    
    def test_get_overlap_info_no_overlap(self, sample_images):
        """Test overlap info with no overlap"""
        processor = SlidingWindowProcessor(window_size=2, overlap=0)
        windows = processor.create_windows(sample_images)
        overlap_info = processor.get_overlap_info()
        
        assert len(overlap_info) == 0  # No overlaps
    
    def test_get_statistics(self, processor, sample_images):
        """Test processor statistics"""
        windows = processor.create_windows(sample_images)
        stats = processor.get_statistics()
        
        assert stats['total_windows'] == len(windows)
        assert stats['total_pages'] == len(sample_images)
        assert stats['window_size'] == processor.window_size
        assert stats['overlap'] == processor.overlap
        assert stats['step_size'] == processor.window_size - processor.overlap
        assert 'windows_info' in stats
    
    def test_get_statistics_empty(self, processor):
        """Test statistics with no windows"""
        stats = processor.get_statistics()
        
        assert stats['total_windows'] == 0
        assert stats['total_pages'] == 0
        assert stats['window_size'] == processor.window_size
        assert stats['overlap'] == processor.overlap
    
    def test_validate_windows(self, processor, sample_images):
        """Test window validation"""
        processor.create_windows(sample_images)
        warnings = processor.validate_windows()
        
        # Should have no warnings for valid configuration
        assert isinstance(warnings, list)
    
    def test_validate_windows_with_gaps(self, processor):
        """Test window validation with gaps (conceptual)"""
        # This would require manipulating windows to create gaps
        # For now, test the empty case
        warnings = processor.validate_windows()
        assert "No windows created" in warnings
    
    def test_validate_windows_inconsistent_overlap(self, processor, sample_images):
        """Test validation with inconsistent overlap"""
        windows = processor.create_windows(sample_images)
        
        # Manually modify a window to create inconsistency
        if len(windows) > 1:
            windows[0].page_numbers = [1]  # Remove overlap
        
        processor.windows = windows
        warnings = processor.validate_windows()
        
        # Should detect inconsistent overlap
        assert any("Inconsistent overlap" in warning for warning in warnings)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_very_large_window_size(self, sample_images):
        """Test with window size larger than available pages"""
        processor = SlidingWindowProcessor(window_size=10, overlap=1)
        windows = processor.create_windows(sample_images)  # Only 6 pages
        
        assert len(windows) == 1  # Should create one window with all pages
        assert len(windows[0].pages) == len(sample_images)
    
    def test_odd_page_counts(self):
        """Test with odd number of pages"""
        # Create 5 test images
        images = []
        for i in range(5):
            img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            images.append(img)
        
        processor = SlidingWindowProcessor(window_size=2, overlap=1)
        windows = processor.create_windows(images)
        
        # Should handle odd number appropriately
        assert len(windows) > 0
        assert all(len(w.pages) >= 1 for w in windows)
    
    def test_memory_efficiency(self, sample_images):
        """Test that processor doesn't hold unnecessary references"""
        processor = SlidingWindowProcessor(window_size=2, overlap=1)
        windows = processor.create_windows(sample_images)
        
        # Clear the processor's windows
        processor.windows = []
        
        # Windows should still be valid (no shared references)
        assert all(len(w.pages) > 0 for w in windows)
    
    @patch('models.window_processor.logger')
    def test_logging(self, mock_logger, processor, sample_images):
        """Test that appropriate logging occurs"""
        processor.create_windows(sample_images)
        
        # Check that info logging was called
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for window processor"""
    
    async def test_full_processing_pipeline(self, sample_images, mock_dolphin_model):
        """Test complete processing pipeline"""
        processor = SlidingWindowProcessor(window_size=2, overlap=1)
        
        # Create windows
        windows = processor.create_windows(sample_images)
        
        # Process windows
        results = await processor.process_windows_async(
            windows=windows,
            dolphin_model=mock_dolphin_model,
            prompt="Process these pages"
        )
        
        # Analyze results
        overlap_info = processor.get_overlap_info()
        stats = processor.get_statistics()
        warnings = processor.validate_windows()
        
        # Verify complete pipeline
        assert len(results) == len(windows)
        assert len(overlap_info) == len(windows) - 1
        assert stats['total_windows'] == len(windows)
        assert isinstance(warnings, list) 
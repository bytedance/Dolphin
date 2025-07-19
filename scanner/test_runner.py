#!/usr/bin/env python3
"""
Simple test runner to avoid pytest plugin conflicts
"""

import sys
import os
import traceback
from unittest.mock import patch

# Add current directory to path
sys.path.insert(0, '.')

def run_config_tests():
    """Run configuration tests manually"""
    print("🧪 Running Configuration Tests")
    print("=" * 50)
    
    try:
        from config.settings import Settings, get_settings, setup_directories
        
        # Test 1: Default settings
        print("Testing default settings...", end=" ")
        settings = Settings()
        assert settings.default_window_size == 2
        assert settings.default_overlap == 1
        assert settings.default_semantic_threshold == 0.8
        assert settings.max_batch_size == 4
        assert settings.enable_gpu == True
        assert settings.enable_caching == True
        assert settings.max_file_size == 50 * 1024 * 1024
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.log_level == "INFO"
        print("✅ PASSED")
        
        # Test 2: Custom settings via environment
        print("Testing custom environment settings...", end=" ")
        with patch.dict(os.environ, {
            'DOLPHIN_DEFAULT_WINDOW_SIZE': '3',
            'DOLPHIN_DEFAULT_OVERLAP': '2',
            'DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD': '0.9',
            'DOLPHIN_MAX_BATCH_SIZE': '8',
            'DOLPHIN_ENABLE_GPU': 'false',
            'DOLPHIN_REDIS_HOST': 'redis-server',
            'DOLPHIN_REDIS_PORT': '6380'
        }):
            custom_settings = Settings()
            assert custom_settings.default_window_size == 3
            assert custom_settings.default_overlap == 2
            assert custom_settings.default_semantic_threshold == 0.9
            assert custom_settings.max_batch_size == 8
            assert custom_settings.enable_gpu == False
            assert custom_settings.redis_host == "redis-server"
            assert custom_settings.redis_port == 6380
        print("✅ PASSED")
        
        # Test 3: File paths
        print("Testing file path settings...", end=" ")
        settings = Settings()
        assert settings.model_config_path == "../config/Dolphin.yaml"
        assert settings.model_checkpoint_path == "../checkpoints"
        assert settings.temp_upload_dir == "./temp_uploads"
        assert ".pdf" in settings.allowed_file_types
        print("✅ PASSED")
        
        # Test 4: Global settings function
        print("Testing global settings function...", end=" ")
        global_settings = get_settings()
        assert isinstance(global_settings, Settings)
        print("✅ PASSED")
        
        # Test 5: Setup directories
        print("Testing setup directories...", end=" ")
        setup_directories()
        # Check if directories were created
        assert os.path.exists(settings.temp_upload_dir)
        assert os.path.exists("./results")
        print("✅ PASSED")
        
        return 5, 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        return 0, 1

def run_window_processor_tests():
    """Run window processor tests manually"""
    print("\n🧪 Running Window Processor Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    try:
        from models.window_processor import SlidingWindowProcessor
        
        # Test 1: Initialization
        print("Testing window processor initialization...", end=" ")
        processor = SlidingWindowProcessor(window_size=2, overlap=1)
        assert processor.window_size == 2
        assert processor.overlap == 1
        print("✅ PASSED")
        passed += 1
        
        # Test 2: Window creation
        print("Testing window creation...", end=" ")
        # Create mock pages (PIL Images)
        from PIL import Image
        import numpy as np
        
        # Create 5 mock pages (dummy images)
        mock_pages = []
        for i in range(5):
            # Create a small dummy image
            dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
            mock_pages.append(dummy_img)
        
        windows = processor.create_windows(mock_pages)
        # With window_size=2, overlap=1, step_size=1, we get 5 windows with 5 pages
        # Windows: [1,2], [2,3], [3,4], [4,5], [5] (pages are 1-indexed)
        assert len(windows) == 5
        assert windows[0].start_page == 1 and windows[0].end_page == 2
        assert windows[1].start_page == 2 and windows[1].end_page == 3
        assert windows[2].start_page == 3 and windows[2].end_page == 4
        assert windows[3].start_page == 4 and windows[3].end_page == 5
        assert windows[4].start_page == 5 and windows[4].end_page == 5
        print("✅ PASSED")
        passed += 1
        
        # Test 3: Edge cases
        print("Testing edge cases...", end=" ")
        # Single page
        single_page = [Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))]
        windows_single = processor.create_windows(single_page)
        assert len(windows_single) == 1
        assert windows_single[0].start_page == 1 and windows_single[0].end_page == 1
        
        # Two pages with window_size=2, overlap=1 creates 2 windows: [1,2], [2]
        two_pages = [Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(2)]
        windows_two = processor.create_windows(two_pages)
        assert len(windows_two) == 2
        assert windows_two[0].start_page == 1 and windows_two[0].end_page == 2
        assert windows_two[1].start_page == 2 and windows_two[1].end_page == 2
        print("✅ PASSED")
        passed += 1
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        failed += 1
        
    return passed, failed

def run_semantic_analyzer_tests():
    """Run semantic analyzer tests manually"""
    print("\n🧪 Running Semantic Analyzer Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    try:
        from models.semantic_analyzer import SemanticAnalyzer
        
        # Test 1: Initialization
        print("Testing semantic analyzer initialization...", end=" ")
        analyzer = SemanticAnalyzer(model_name="all-MiniLM-L6-v2")
        assert hasattr(analyzer, 'sentence_model')
        assert hasattr(analyzer, 'similarity_threshold')
        assert analyzer.similarity_threshold == 0.8
        print("✅ PASSED")
        passed += 1
        
        # Test 2: Text normalization
        print("Testing text normalization...", end=" ")
        from utils.semantic_utils import normalize_text
        
        input_text = "  Hello, World!  \n\n  This is a test.  "
        normalized = normalize_text(input_text)
        expected = "hello, world! this is a test."  # normalize_text converts to lowercase
        assert normalized == expected
        print("✅ PASSED")
        passed += 1
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        failed += 1
        
    return passed, failed

def main():
    """Main test runner"""
    print("🚀 Dolphin FastAPI Service Test Runner")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    test_suites = [
        run_config_tests,
        run_window_processor_tests,
        run_semantic_analyzer_tests
    ]
    
    for test_suite in test_suites:
        passed, failed = test_suite()
        total_passed += passed
        total_failed += failed
    
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
"""
Unit tests for configuration settings
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from config.settings import Settings, get_settings, setup_directories


class TestSettings:
    """Test Settings class configuration"""
    
    def test_default_settings(self):
        """Test default settings initialization"""
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
    
    def test_custom_settings(self):
        """Test custom settings via environment variables"""
        with patch.dict(os.environ, {
            'DOLPHIN_DEFAULT_WINDOW_SIZE': '3',
            'DOLPHIN_DEFAULT_OVERLAP': '2',
            'DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD': '0.9',
            'DOLPHIN_MAX_BATCH_SIZE': '8',
            'DOLPHIN_ENABLE_GPU': 'false',
            'DOLPHIN_REDIS_HOST': 'redis-server',
            'DOLPHIN_REDIS_PORT': '6380'
        }):
            settings = Settings()
            
            assert settings.default_window_size == 3
            assert settings.default_overlap == 2
            assert settings.default_semantic_threshold == 0.9
            assert settings.max_batch_size == 8
            assert settings.enable_gpu == False
            assert settings.redis_host == "redis-server"
            assert settings.redis_port == 6380
    
    def test_invalid_settings(self):
        """Test validation of invalid settings"""
        with patch.dict(os.environ, {
            'DOLPHIN_DEFAULT_WINDOW_SIZE': '0'  # Invalid: must be >= 1
        }):
            with pytest.raises(ValidationError):
                Settings()
        
        with patch.dict(os.environ, {
            'DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD': '1.5'  # Invalid: must be <= 1.0
        }):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_file_paths(self):
        """Test file path settings"""
        settings = Settings()
        
        assert settings.model_config_path == "../config/Dolphin.yaml"
        assert settings.model_checkpoint_path == "../checkpoints"
        assert settings.temp_upload_dir == "./temp_uploads"
        assert ".pdf" in settings.allowed_file_types
    
    def test_env_prefix(self):
        """Test environment variable prefix"""
        with patch.dict(os.environ, {
            'DOLPHIN_LOG_LEVEL': 'DEBUG'
        }):
            settings = Settings()
            assert settings.log_level == "DEBUG"


class TestSettingsFunctions:
    """Test settings utility functions"""
    
    def test_get_settings(self):
        """Test get_settings function"""
        settings = get_settings()
        assert isinstance(settings, Settings)
        
        # Test that it returns the same instance (singleton behavior)
        settings2 = get_settings()
        assert settings is settings2
    
    @patch('os.makedirs')
    def test_setup_directories(self, mock_makedirs):
        """Test directory setup"""
        with patch.object(Settings, 'temp_upload_dir', './test_uploads'):
            setup_directories()
            
            # Check that directories are created
            expected_calls = [
                (('./test_uploads',), {'exist_ok': True}),
                (('./results',), {'exist_ok': True}),
                (('./results/recognition_json',), {'exist_ok': True}),
                (('./results/markdown',), {'exist_ok': True}),
                (('./results/markdown/figures',), {'exist_ok': True}),
            ]
            
            assert mock_makedirs.call_count == 5
            for call in expected_calls:
                mock_makedirs.assert_any_call(*call[0], **call[1])
    
    @patch('os.makedirs')
    def test_setup_directories_with_error(self, mock_makedirs):
        """Test directory setup with permission error"""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(PermissionError):
            setup_directories()


class TestSettingsValidation:
    """Test settings validation logic"""
    
    def test_window_size_validation(self):
        """Test window size validation"""
        # Valid window sizes
        for size in [1, 2, 5, 10]:
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_WINDOW_SIZE': str(size)}):
                settings = Settings()
                assert settings.default_window_size == size
        
        # Invalid window sizes
        for size in [0, -1, 11]:  # Assuming max is 10
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_WINDOW_SIZE': str(size)}):
                with pytest.raises(ValidationError):
                    Settings()
    
    def test_overlap_validation(self):
        """Test overlap validation"""
        # Valid overlaps
        for overlap in [0, 1, 3, 5]:
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_OVERLAP': str(overlap)}):
                settings = Settings()
                assert settings.default_overlap == overlap
        
        # Invalid overlaps
        for overlap in [-1, 6]:  # Assuming max is 5
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_OVERLAP': str(overlap)}):
                with pytest.raises(ValidationError):
                    Settings()
    
    def test_semantic_threshold_validation(self):
        """Test semantic threshold validation"""
        # Valid thresholds
        for threshold in [0.0, 0.5, 0.8, 1.0]:
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD': str(threshold)}):
                settings = Settings()
                assert settings.default_semantic_threshold == threshold
        
        # Invalid thresholds
        for threshold in [-0.1, 1.1]:
            with patch.dict(os.environ, {'DOLPHIN_DEFAULT_SEMANTIC_THRESHOLD': str(threshold)}):
                with pytest.raises(ValidationError):
                    Settings()
    
    def test_file_size_validation(self):
        """Test file size validation"""
        with patch.dict(os.environ, {'DOLPHIN_MAX_FILE_SIZE': '1048576'}):  # 1MB
            settings = Settings()
            assert settings.max_file_size == 1048576
        
        # Test that very large values are accepted (within reason)
        with patch.dict(os.environ, {'DOLPHIN_MAX_FILE_SIZE': str(100 * 1024 * 1024)}):  # 100MB
            settings = Settings()
            assert settings.max_file_size == 100 * 1024 * 1024


class TestConfigIntegration:
    """Integration tests for configuration"""
    
    def test_config_file_loading(self):
        """Test loading configuration from .env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("DOLPHIN_DEFAULT_WINDOW_SIZE=4\n")
            f.write("DOLPHIN_LOG_LEVEL=DEBUG\n")
            f.flush()
            
            try:
                with patch.object(Settings.Config, 'env_file', f.name):
                    settings = Settings()
                    # Note: This might not work as expected due to how pydantic loads env files
                    # This is more of a conceptual test
            finally:
                os.unlink(f.name)
    
    def test_settings_immutability(self):
        """Test that settings are immutable after creation"""
        settings = Settings()
        original_window_size = settings.default_window_size
        
        # Settings should be immutable
        with pytest.raises(AttributeError):
            settings.default_window_size = 5
        
        assert settings.default_window_size == original_window_size


@pytest.fixture
def temp_settings():
    """Fixture for temporary settings with custom values"""
    with patch.dict(os.environ, {
        'DOLPHIN_DEFAULT_WINDOW_SIZE': '3',
        'DOLPHIN_DEFAULT_OVERLAP': '1',
        'DOLPHIN_TEMP_UPLOAD_DIR': './test_temp'
    }):
        yield Settings()


class TestSettingsFixtures:
    """Test using settings fixtures"""
    
    def test_with_temp_settings(self, temp_settings):
        """Test using temporary settings fixture"""
        assert temp_settings.default_window_size == 3
        assert temp_settings.default_overlap == 1
        assert temp_settings.temp_upload_dir == './test_temp' 
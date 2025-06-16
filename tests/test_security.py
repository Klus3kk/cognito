# tests/test_security.py
import pytest
import tempfile
import os
import time
from src.utils.sanitizer import InputSanitizer, SecurityValidator
from src.utils.rate_limiter import RateLimiter

class TestInputSanitization:
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
        self.validator = SecurityValidator()
    
    def test_code_sanitization(self):
        """Test code input sanitization."""
        # Test normal code
        normal_code = "def hello():\n    print('world')"
        sanitized = self.sanitizer.sanitize_code(normal_code)
        assert sanitized == normal_code
        
        # Test with null bytes
        malicious_code = "def hello():\n    print('world\x00')"
        sanitized = self.sanitizer.sanitize_code(malicious_code)
        assert '\x00' not in sanitized
        
        # Test length limit
        with pytest.raises(ValueError, match="Code too long"):
            self.sanitizer.sanitize_code("x" * 100001)
        
        # Test line length limiting
        long_line_code = "x" * 15000
        sanitized = self.sanitizer.sanitize_code(long_line_code)
        assert "truncated for safety" in sanitized
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        # Normal filename
        assert self.sanitizer.sanitize_filename("test.py") == "test.py"
        
        # Path traversal attempt
        assert self.sanitizer.sanitize_filename("../../../etc/passwd") == "passwd"
        
        # Windows path
        assert self.sanitizer.sanitize_filename("C:\\Windows\\test.py") == "test.py"
        
        # Dangerous characters
        assert "<>" not in self.sanitizer.sanitize_filename("test<script>.py")
        
        # Reserved names
        assert self.sanitizer.sanitize_filename("CON") == "file_CON"
        
        # Empty filename
        assert self.sanitizer.sanitize_filename("") == "untitled"
    
    def test_dangerous_pattern_detection(self):
        """Test detection of dangerous patterns."""
        safe_code = "def add(a, b):\n    return a + b"
        dangerous_code = "eval('malicious_code')"
        
        assert len(self.sanitizer.check_dangerous_patterns(safe_code)) == 0
        assert len(self.sanitizer.check_dangerous_patterns(dangerous_code)) > 0
    
    def test_file_upload_validation(self):
        """Test file upload security validation."""
        # Valid text file
        valid_content = b"def hello():\n    print('world')"
        result = self.validator.validate_upload(valid_content, "test.py")
        assert result['valid'] is True
        
        # File too large
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        result = self.validator.validate_upload(large_content, "large.py")
        assert result['valid'] is False
        assert "too large" in result['issues'][0]
        
        # Binary file
        binary_content = b"def hello():\x00\x01\x02"
        result = self.validator.validate_upload(binary_content, "binary.py")
        assert result['valid'] is False
        assert "Binary content" in result['issues'][0]

class TestRateLimiting:
    
    def setup_method(self):
        self.limiter = RateLimiter()
    
    def test_sliding_window_rate_limit(self):
        """Test sliding window rate limiting."""
        client_id = "test_client"
        
        # Should allow first request
        allowed, info = self.limiter.check_rate_limit(client_id, limit=2, window=1)
        assert allowed is True
        assert info['current'] == 1
        
        # Should allow second request
        allowed, info = self.limiter.check_rate_limit(client_id, limit=2, window=1)
        assert allowed is True
        assert info['current'] == 2
        
        # Should block third request
        allowed, info = self.limiter.check_rate_limit(client_id, limit=2, window=1)
        assert allowed is False
        assert info['current'] == 2
        
        # Wait for window to pass and try again
        time.sleep(1.1)
        allowed, info = self.limiter.check_rate_limit(client_id, limit=2, window=1)
        assert allowed is True
    
    def test_token_bucket_rate_limit(self):
        """Test token bucket rate limiting."""
        client_id = "test_client"
        
        # Should allow requests up to capacity
        for i in range(5):
            allowed, info = self.limiter.check_token_bucket(client_id, capacity=5, refill_rate=1.0)
            assert allowed is True
        
        # Should block when bucket is empty
        allowed, info = self.limiter.check_token_bucket(client_id, capacity=5, refill_rate=1.0)
        assert allowed is False
    
    def test_client_stats(self):
        """Test client statistics tracking."""
        client_id = "test_client"
        
        # Make some requests
        for i in range(3):
            self.limiter.check_rate_limit(client_id, limit=10, window=3600)
        
        stats = self.limiter.get_client_stats(client_id)
        assert stats['requests_last_hour'] == 3
        assert stats['total_requests'] == 3

# tests/test_performance.py
import pytest
import time
from src.utils.performance import LRUCache, AnalysisCache, ChunkProcessor, cached_analysis

class TestPerformanceOptimization:
    
    def test_lru_cache(self):
        """Test LRU cache implementation."""
        cache = LRUCache(maxsize=2)
        
        # Add items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # Add third item, should evict least recently used
        cache.put("key3", "value3")
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_analysis_cache_ttl(self):
        """Test analysis cache with TTL."""
        cache = AnalysisCache(maxsize=10, ttl_seconds=1)
        
        # Add item
        cache.put("test_hash", {"result": "test"})
        assert cache.get("test_hash") == {"result": "test"}
        
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_hash") is None
    
    def test_chunk_processor(self):
        """Test large file processing in chunks."""
        processor = ChunkProcessor(chunk_size=100)
        
        # Create large content
        large_content = "line\n" * 50  # 250 characters
        
        def simple_processor(chunk):
            return {"lines": len(chunk.split('\n'))}
        
        results = processor.process_large_file(large_content, simple_processor)
        assert len(results) >= 2  # Should be split into multiple chunks
        
        merged = processor.merge_results(results)
        assert "lines" in merged
    
    def test_cached_analysis_decorator(self):
        """Test cached analysis decorator."""
        call_count = 0
        
        @cached_analysis(cache_ttl=60)
        def dummy_analysis(code):
            nonlocal call_count
            call_count += 1
            return {"analysis": f"result_{call_count}"}
        
        # First call should execute function
        result1 = dummy_analysis("test code")
        assert call_count == 1
        
        # Second call with same code should use cache
        result2 = dummy_analysis("test code")
        assert call_count == 1
        assert result1 == result2
        
        # Different code should execute function again
        result3 = dummy_analysis("different code")
        assert call_count == 2

# tests/test_configuration.py
import pytest
import os
from src.config import CognitoConfig, get_config, reload_config

class TestConfiguration:
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = CognitoConfig()
        assert config.environment == "development"
        assert config.debug is True
        assert config.analysis.max_file_size_mb == 10
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["COGNITO_ENV"] = "production"
        os.environ["COGNITO_MAX_FILE_SIZE_MB"] = "20"
        os.environ["OPENAI_API_KEY"] = "test_key"
        
        try:
            config = CognitoConfig()
            assert config.environment == "production"
            assert config.analysis.max_file_size_mb == 20
            assert config.llm.openai_api_key == "test_key"
        finally:
            # Clean up
            for key in ["COGNITO_ENV", "COGNITO_MAX_FILE_SIZE_MB", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)
    
    def test_production_overrides(self):
        """Test production environment overrides."""
        os.environ["COGNITO_ENV"] = "production"
        
        try:
            config = CognitoConfig()
            assert config.debug is False
            assert config.logging.level == "WARNING"
            assert config.security.rate_limit_enabled is True
        finally:
            os.environ.pop("COGNITO_ENV", None)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = CognitoConfig()
        config.analysis.max_file_size_mb = -1  # Invalid value
        
        issues = config.validate()
        assert len(issues) > 0
        assert any("max_file_size_mb must be positive" in issue for issue in issues)
    
    def test_llm_enabled_check(self):
        """Test LLM enabled check."""
        config = CognitoConfig()
        
        # Without API key
        config.llm.openai_api_key = None
        assert config.get_llm_enabled() is False
        
        # With API key
        config.llm.openai_api_key = "test_key"
        assert config.get_llm_enabled() is True

# tests/test_error_handling.py
import pytest
from src.utils.error_handling import CognitoException, AnalysisError, SecurityError, ErrorHandler

class TestErrorHandling:
    
    def test_cognito_exception_creation(self):
        """Test creation of Cognito exceptions."""
        error = AnalysisError("Test error", details="Test details")
        assert error.error.message == "Test error"
        assert error.error.details == "Test details"
        assert error.error.category.value == "analysis"
    
    def test_error_handler_logging(self):
        """Test error handler logging."""
        handler = ErrorHandler()
        
        error = SecurityError("Security issue")
        logged_error = handler.log_error(error)
        
        assert logged_error.message == "Security issue"
        assert len(handler.recent_errors) == 1
    
    def test_error_statistics(self):
        """Test error statistics tracking."""
        handler = ErrorHandler()
        
        # Log different types of errors
        handler.log_error(AnalysisError("Analysis error 1"))
        handler.log_error(AnalysisError("Analysis error 2"))
        handler.log_error(SecurityError("Security error"))
        
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 3
        assert stats['errors_by_category']['analysis'] == 2
        assert stats['errors_by_category']['security'] == 1

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
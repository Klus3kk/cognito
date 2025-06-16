"""
Input validation and sanitization utilities for Cognito.
Ensures security and prevents malicious input.
"""

import re
import os
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

from config import get_config
from utils.error_handling import SecurityError, FileProcessingError, validate_input


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CodeValidator:
    """Validates and sanitizes code input."""
    
    def __init__(self):
        self.config = get_config()
        
        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r'__import__\s*\(',  # Dynamic imports
            r'exec\s*\(',        # Code execution
            r'eval\s*\(',        # Expression evaluation
            r'compile\s*\(',     # Code compilation
            r'globals\s*\(',     # Global namespace access
            r'locals\s*\(',      # Local namespace access
            r'vars\s*\(',        # Variable access
            r'dir\s*\(',         # Directory listing
            r'getattr\s*\(',     # Attribute access
            r'setattr\s*\(',     # Attribute setting
            r'delattr\s*\(',     # Attribute deletion
            r'hasattr\s*\(',     # Attribute checking
        ]
        
        # Suspicious patterns (warnings, not errors)
        self.suspicious_patterns = [
            r'os\.system\s*\(',     # System commands
            r'subprocess\.',        # Subprocess calls
            r'open\s*\(',          # File operations
            r'input\s*\(',         # User input
            r'raw_input\s*\(',     # Raw user input (Python 2)
        ]
    
    def validate_code_content(self, code: str) -> ValidationResult:
        """Validate code content for security and safety."""
        result = ValidationResult(is_valid=True, sanitized_value=code)
        
        # Check code length
        if len(code) > self.config.security.max_code_length:
            result.is_valid = False
            result.errors.append(
                f"Code too long: {len(code)} characters (max: {self.config.security.max_code_length})"
            )
            return result
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                if self.config.security.enable_input_validation:
                    result.is_valid = False
                    result.errors.append(f"Dangerous pattern detected: {pattern}")
                else:
                    result.warnings.append(f"Potentially dangerous pattern: {pattern}")
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                result.warnings.append(f"Suspicious pattern detected: {pattern}")
        
        # Basic encoding validation
        try:
            code.encode('utf-8')
        except UnicodeEncodeError as e:
            result.is_valid = False
            result.errors.append(f"Invalid UTF-8 encoding: {str(e)}")
        
        # Check for null bytes
        if '\x00' in code:
            result.is_valid = False
            result.errors.append("Null bytes detected in code")
        
        # Sanitize the code if valid
        if result.is_valid:
            result.sanitized_value = self._sanitize_code(code)
        
        return result
    
    def _sanitize_code(self, code: str) -> str:
        """Sanitize code input."""
        # Remove null bytes
        code = code.replace('\x00', '')
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Limit line length to prevent buffer overflow attacks
        lines = code.split('\n')
        sanitized_lines = []
        max_line_length = 10000  # 10KB per line max
        
        for line in lines:
            if len(line) > max_line_length:
                # Truncate very long lines
                sanitized_lines.append(line[:max_line_length] + '\n# ... (line truncated)')
            else:
                sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)


class FileValidator:
    """Validates file uploads and paths."""
    
    def __init__(self):
        self.config = get_config()
    
    def validate_file_path(self, file_path: str) -> ValidationResult:
        """Validate file path for security."""
        result = ValidationResult(is_valid=True, sanitized_value=file_path)
        
        # Normalize path
        try:
            normalized_path = os.path.normpath(file_path)
            resolved_path = os.path.abspath(normalized_path)
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Invalid file path: {str(e)}")
            return result
        
        # Check for path traversal attacks
        if '..' in file_path or file_path.startswith('/'):
            result.is_valid = False
            result.errors.append("Path traversal attempt detected")
            return result
        
        # Validate file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.config.security.allowed_file_extensions:
            result.is_valid = False
            result.errors.append(f"File extension not allowed: {file_ext}")
            return result
        
        result.sanitized_value = resolved_path
        return result
    
    def validate_file_content(self, file_path: str, content: bytes) -> ValidationResult:
        """Validate file content."""
        result = ValidationResult(is_valid=True, sanitized_value=content)
        
        # Check file size
        file_size = len(content)
        max_size = self.config.analysis.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            result.is_valid = False
            result.errors.append(
                f"File too large: {file_size / 1024 / 1024:.1f} MB "
                f"(max: {self.config.analysis.max_file_size_mb} MB)"
            )
            return result
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and not mime_type.startswith('text/'):
            result.warnings.append(f"Non-text MIME type detected: {mime_type}")
        
        # Check for binary content
        if b'\x00' in content:
            result.is_valid = False
            result.errors.append("Binary content detected (null bytes found)")
            return result
        
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
            result.sanitized_value = text_content
        except UnicodeDecodeError:
            try:
                # Try common encodings
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    text_content = content.decode(encoding)
                    result.sanitized_value = text_content
                    result.warnings.append(f"File decoded using {encoding} encoding")
                    break
            except UnicodeDecodeError:
                result.is_valid = False
                result.errors.append("Unable to decode file as text")
                return result
        
        return result


class ParameterValidator:
    """Validates API parameters and user input."""
    
    def validate_language(self, language: str) -> ValidationResult:
        """Validate programming language parameter."""
        result = ValidationResult(is_valid=True, sanitized_value=language.lower().strip())
        
        config = get_config()
        
        # Basic sanitization
        sanitized = re.sub(r'[^a-zA-Z0-9+#-]', '', language.lower().strip())
        
        # Check if language is supported
        if sanitized not in config.analysis.supported_languages:
            result.warnings.append(f"Language '{sanitized}' not in supported list, will use generic analysis")
        
        result.sanitized_value = sanitized
        return result
    
    def validate_boolean_parameter(self, value: Any, param_name: str) -> ValidationResult:
        """Validate boolean parameter."""
        result = ValidationResult(is_valid=True)
        
        if isinstance(value, bool):
            result.sanitized_value = value
        elif isinstance(value, str):
            lower_val = value.lower().strip()
            if lower_val in ['true', '1', 'yes', 'on']:
                result.sanitized_value = True
            elif lower_val in ['false', '0', 'no', 'off']:
                result.sanitized_value = False
            else:
                result.is_valid = False
                result.errors.append(f"Invalid boolean value for {param_name}: {value}")
        else:
            result.is_valid = False
            result.errors.append(f"Invalid type for boolean parameter {param_name}: {type(value)}")
        
        return result
    
    def validate_integer_parameter(
        self, 
        value: Any, 
        param_name: str, 
        min_value: Optional[int] = None, 
        max_value: Optional[int] = None
    ) -> ValidationResult:
        """Validate integer parameter."""
        result = ValidationResult(is_valid=True)
        
        try:
            int_value = int(value)
            
            if min_value is not None and int_value < min_value:
                result.is_valid = False
                result.errors.append(f"{param_name} must be >= {min_value}, got {int_value}")
                return result
            
            if max_value is not None and int_value > max_value:
                result.is_valid = False
                result.errors.append(f"{param_name} must be <= {max_value}, got {int_value}")
                return result
            
            result.sanitized_value = int_value
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f"Invalid integer value for {param_name}: {value}")
        
        return result
    
    def validate_string_parameter(
        self, 
        value: Any, 
        param_name: str, 
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None
    ) -> ValidationResult:
        """Validate string parameter."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(value, str):
            result.is_valid = False
            result.errors.append(f"{param_name} must be a string, got {type(value)}")
            return result
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Check length
        if max_length and len(sanitized) > max_length:
            result.is_valid = False
            result.errors.append(f"{param_name} too long: {len(sanitized)} chars (max: {max_length})")
            return result
        
        # Check allowed characters
        if allowed_chars:
            if not re.match(f'^[{re.escape(allowed_chars)}]*$', sanitized):
                result.is_valid = False
                result.errors.append(f"{param_name} contains invalid characters")
                return result
        
        result.sanitized_value = sanitized
        return result


class RateLimiter:
    """Rate limiting for API requests."""
    
    def __init__(self):
        self.request_counts = {}
        self.config = get_config()
    
    def check_rate_limit(self, client_id: str) -> ValidationResult:
        """Check if client has exceeded rate limit."""
        result = ValidationResult(is_valid=True)
        
        if not self.config.security.rate_limit_enabled:
            return result
        
        import time
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour
        
        # Clean old entries
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if timestamp > hour_ago
            ]
        else:
            self.request_counts[client_id] = []
        
        # Check current count
        current_count = len(self.request_counts[client_id])
        
        if current_count >= self.config.security.rate_limit_per_hour:
            result.is_valid = False
            result.errors.append(
                f"Rate limit exceeded: {current_count}/{self.config.security.rate_limit_per_hour} "
                f"requests in the last hour"
            )
            return result
        
        # Record this request
        self.request_counts[client_id].append(current_time)
        return result


# Global validators
_code_validator: Optional[CodeValidator] = None
_file_validator: Optional[FileValidator] = None
_parameter_validator: Optional[ParameterValidator] = None
_rate_limiter: Optional[RateLimiter] = None


def get_code_validator() -> CodeValidator:
    """Get the global code validator."""
    global _code_validator
    if _code_validator is None:
        _code_validator = CodeValidator()
    return _code_validator


def get_file_validator() -> FileValidator:
    """Get the global file validator."""
    global _file_validator
    if _file_validator is None:
        _file_validator = FileValidator()
    return _file_validator


def get_parameter_validator() -> ParameterValidator:
    """Get the global parameter validator."""
    global _parameter_validator
    if _parameter_validator is None:
        _parameter_validator = ParameterValidator()
    return _parameter_validator


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# Convenience functions
def validate_code(code: str) -> str:
    """Validate and sanitize code input."""
    result = get_code_validator().validate_code_content(code)
    
    if not result.is_valid:
        raise SecurityError(
            "Code validation failed",
            details="; ".join(result.errors),
            error_code="CODE_VALIDATION_FAILED"
        )
    
    return result.sanitized_value


def validate_file(file_path: str, content: Union[bytes, str] = None) -> tuple:
    """Validate file path and content."""
    file_validator = get_file_validator()
    
    # Validate path
    path_result = file_validator.validate_file_path(file_path)
    if not path_result.is_valid:
        raise FileProcessingError(
            "File path validation failed",
            details="; ".join(path_result.errors),
            error_code="INVALID_FILE_PATH"
        )
    
    # Validate content if provided
    if content is not None:
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        content_result = file_validator.validate_file_content(file_path, content)
        if not content_result.is_valid:
            raise FileProcessingError(
                "File content validation failed",
                details="; ".join(content_result.errors),
                error_code="INVALID_FILE_CONTENT"
            )
        
        return path_result.sanitized_value, content_result.sanitized_value
    
    return path_result.sanitized_value, None


def check_rate_limit(client_id: str) -> None:
    """Check rate limit for client."""
    result = get_rate_limiter().check_rate_limit(client_id)
    
    if not result.is_valid:
        from utils.error_handling import LLMAPIError
        raise LLMAPIError(
            "Rate limit exceeded",
            details="; ".join(result.errors),
            error_code="RATE_LIMIT_EXCEEDED"
        )
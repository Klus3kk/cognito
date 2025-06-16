"""
Input sanitization and security validation for Cognito.
"""

import re
import html
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from config import get_config


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    is_safe: bool
    sanitized_value: Any
    warnings: List[str]
    blocked_patterns: List[str]


class InputSanitizer:
    """Sanitizes and validates user input."""
    
    def __init__(self):
        self.config = get_config().security
        
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = {
            'code_injection': [
                r'__import__\s*\(',
                r'exec\s*\(',
                r'eval\s*\(',
                r'compile\s*\(',
                r'globals\s*\(',
                r'locals\s*\(',
                r'vars\s*\(',
                r'dir\s*\(',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'delattr\s*\(',
                r'hasattr\s*\(',
            ],
            'system_access': [
                r'os\.system\s*\(',
                r'subprocess\.',
                r'popen\s*\(',
                r'spawn\s*\(',
                r'fork\s*\(',
            ],
            'file_access': [
                r'open\s*\(\s*["\'][^"\']*\.\./[^"\']*["\']',  # Path traversal
                r'\.\./',
                r'/etc/',
                r'/proc/',
                r'/sys/',
                r'C:\\',
                r'\\\\',
            ],
            'script_injection': [
                r'<script.*?>',
                r'javascript:',
                r'vbscript:',
                r'on\w+\s*=',
            ],
            'sql_injection': [
                r';\s*(drop|delete|insert|update|create|alter)',
                r'union\s+select',
                r'0x[0-9a-fA-F]+',
                r'char\s*\(',
            ]
        }
        
        # Suspicious but not necessarily dangerous patterns
        self.suspicious_patterns = {
            'network': [
                r'requests\.',
                r'urllib\.',
                r'socket\.',
                r'http\.',
                r'ftp\.',
            ],
            'crypto': [
                r'hashlib\.',
                r'crypto\.',
                r'ssl\.',
                r'random\.',
            ]
        }
    
    def sanitize_code(self, code: str) -> SanitizationResult:
        """Sanitize code input."""
        warnings = []
        blocked_patterns = []
        
        # Check length
        if len(code) > self.config.max_code_length:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"Code too long: {len(code)} chars > {self.config.max_code_length}"],
                blocked_patterns=[]
            )
        
        # Check for null bytes
        if '\x00' in code:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=["Null bytes detected"],
                blocked_patterns=["null_bytes"]
            )
        
        # Check dangerous patterns
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    blocked_patterns.append(f"{category}:{pattern}")
        
        if blocked_patterns and self.config.enable_input_validation:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"Dangerous patterns detected: {len(blocked_patterns)}"],
                blocked_patterns=blocked_patterns
            )
        
        # Check suspicious patterns
        for category, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    warnings.append(f"Suspicious {category} pattern detected")
        
        # Sanitize the code
        sanitized_code = self._sanitize_text(code)
        
        return SanitizationResult(
            is_safe=True,
            sanitized_value=sanitized_code,
            warnings=warnings,
            blocked_patterns=[]
        )
    
    def sanitize_filename(self, filename: str) -> SanitizationResult:
        """Sanitize filename input."""
        warnings = []
        blocked_patterns = []
        
        # Remove path components for security
        filename = Path(filename).name
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            if char in filename:
                blocked_patterns.append(f"dangerous_char:{char}")
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        if ext and ext not in self.config.allowed_file_extensions:
            blocked_patterns.append(f"disallowed_extension:{ext}")
        
        # Check for path traversal
        if '..' in filename or filename.startswith('/') or ':' in filename:
            blocked_patterns.append("path_traversal")
        
        if blocked_patterns:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"Dangerous filename: {filename}"],
                blocked_patterns=blocked_patterns
            )
        
        # Sanitize filename
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        sanitized = re.sub(r'_{2,}', '_', sanitized)  # Replace multiple underscores
        
        if sanitized != filename:
            warnings.append("Filename was sanitized")
        
        return SanitizationResult(
            is_safe=True,
            sanitized_value=sanitized,
            warnings=warnings,
            blocked_patterns=[]
        )
    
    def sanitize_file_content(self, content: bytes, filename: str = "") -> SanitizationResult:
        """Sanitize file content."""
        warnings = []
        blocked_patterns = []
        
        # Check file size
        max_size = self.config.get('max_file_size_mb', 10) * 1024 * 1024
        if len(content) > max_size:
            return SanitizationResult(
                is_safe=False,
                sanitized_value=b"",
                warnings=[f"File too large: {len(content)} bytes"],
                blocked_patterns=["file_too_large"]
            )
        
        # Check for binary content
        if b'\x00' in content:
            blocked_patterns.append("binary_content")
        
        # Check MIME type
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and not mime_type.startswith('text/'):
                warnings.append(f"Non-text MIME type: {mime_type}")
        
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
            # Run code sanitization on text content
            code_result = self.sanitize_code(text_content)
            
            return SanitizationResult(
                is_safe=code_result.is_safe and not blocked_patterns,
                sanitized_value=code_result.sanitized_value.encode('utf-8') if code_result.is_safe else b"",
                warnings=warnings + code_result.warnings,
                blocked_patterns=blocked_patterns + code_result.blocked_patterns
            )
        
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                try:
                    text_content = content.decode(encoding)
                    warnings.append(f"Decoded with {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return SanitizationResult(
                    is_safe=False,
                    sanitized_value=b"",
                    warnings=["Unable to decode file as text"],
                    blocked_patterns=["encoding_error"]
                )
        
        return SanitizationResult(
            is_safe=not blocked_patterns,
            sanitized_value=content,
            warnings=warnings,
            blocked_patterns=blocked_patterns
        )
    
    def sanitize_parameter(self, value: Any, param_name: str, max_length: int = 1000) -> SanitizationResult:
        """Sanitize API parameter."""
        warnings = []
        blocked_patterns = []
        
        # Convert to string if needed
        if not isinstance(value, str):
            value = str(value)
        
        # Check length
        if len(value) > max_length:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"Parameter {param_name} too long: {len(value)} > {max_length}"],
                blocked_patterns=["parameter_too_long"]
            )
        
        # HTML encode to prevent XSS
        sanitized = html.escape(value)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Check for suspicious patterns in parameters
        for pattern in ['<script', 'javascript:', 'data:', 'vbscript:']:
            if pattern.lower() in value.lower():
                blocked_patterns.append(f"script_injection:{pattern}")
        
        if blocked_patterns:
            return SanitizationResult(
                is_safe=False,
                sanitized_value="",
                warnings=[f"Dangerous content in {param_name}"],
                blocked_patterns=blocked_patterns
            )
        
        if sanitized != value:
            warnings.append(f"Parameter {param_name} was sanitized")
        
        return SanitizationResult(
            is_safe=True,
            sanitized_value=sanitized,
            warnings=warnings,
            blocked_patterns=[]
        )
    
    def _sanitize_text(self, text: str) -> str:
        """Basic text sanitization."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Limit line length
        lines = text.split('\n')
        sanitized_lines = []
        
        for line in lines:
            if len(line) > 10000:  # 10KB per line max
                sanitized_lines.append(line[:10000] + '\n# ... (line truncated)')
            else:
                sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)


class SecurityHeaders:
    """Generate security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get standard security headers."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'none'; object-src 'none';",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    @staticmethod
    def get_cors_headers(origin: str = "*") -> Dict[str, str]:
        """Get CORS headers."""
        return {
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
            'Access-Control-Max-Age': '86400'
        }


# Global sanitizer instance
_sanitizer: Optional[InputSanitizer] = None


def get_sanitizer() -> InputSanitizer:
    """Get the global input sanitizer."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer()
    return _sanitizer


def sanitize_code_input(code: str) -> str:
    """Convenience function to sanitize code input."""
    result = get_sanitizer().sanitize_code(code)
    if not result.is_safe:
        from utils.error_handling import SecurityError
        raise SecurityError(
            f"Code input validation failed: {'; '.join(result.warnings)}",
            details=f"Blocked patterns: {result.blocked_patterns}"
        )
    return result.sanitized_value
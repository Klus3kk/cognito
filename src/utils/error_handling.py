"""
Comprehensive error handling and logging utilities for Cognito.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better tracking."""
    USER_INPUT = "user_input"
    ANALYSIS = "analysis"
    LANGUAGE_DETECTION = "language_detection"
    MODEL_LOADING = "model_loading"
    LLM_API = "llm_api"
    FILE_IO = "file_io"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class CognitoError:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    suggestions: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'suggestions': self.suggestions,
            'error_code': self.error_code,
            'timestamp': self.timestamp
        }


class CognitoException(Exception):
    """Base exception for Cognito-specific errors."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        suggestions: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.error = CognitoError(
            category=category,
            severity=severity,
            message=message,
            details=details,
            suggestions=suggestions,
            error_code=error_code
        )


class AnalysisError(CognitoException):
    """Error during code analysis."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.ANALYSIS,
            **kwargs
        )


class LanguageDetectionError(CognitoException):
    """Error during language detection."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.LANGUAGE_DETECTION,
            **kwargs
        )


class ModelLoadingError(CognitoException):
    """Error during model loading."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.MODEL_LOADING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class LLMAPIError(CognitoException):
    """Error during LLM API calls."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.LLM_API,
            **kwargs
        )


class FileProcessingError(CognitoException):
    """Error during file processing."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.FILE_IO,
            **kwargs
        )


class SecurityError(CognitoException):
    """Security-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ErrorHandler:
    """Central error handling and logging system."""
    
    def __init__(self, logger_name: str = "cognito"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def log_error(self, error: Union[CognitoError, Exception, str], context: Optional[Dict[str, Any]] = None):
        """Log an error with proper categorization."""
        if isinstance(error, str):
            error = CognitoError(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                message=error
            )
        elif isinstance(error, CognitoException):
            error = error.error
        elif isinstance(error, Exception):
            error = CognitoError(
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                details=traceback.format_exc()
            )
        
        # Add to recent errors
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Count errors by category
        category_key = error.category.value
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        
        # Log based on severity
        log_data = error.to_dict()
        if context:
            log_data['context'] = context
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=log_data)
        else:
            self.logger.info(f"LOW SEVERITY: {error.message}", extra=log_data)
        
        return error
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': len(self.recent_errors),
            'errors_by_category': self.error_counts.copy(),
            'recent_errors_count': len(self.recent_errors),
            'most_common_category': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors."""
        return [error.to_dict() for error in self.recent_errors[-limit:]]


# Global error handler
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def log_error(error: Union[CognitoError, Exception, str], context: Optional[Dict[str, Any]] = None):
    """Convenience function to log an error."""
    return get_error_handler().log_error(error, context)


def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
    error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error = CognitoError(
            category=error_category,
            severity=error_severity,
            message=f"Error in {func.__name__}: {str(e)}",
            details=traceback.format_exc()
        )
        log_error(error, context)
        return fallback_value


def error_handler(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    fallback_value: Any = None,
    reraise: bool = False
):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = CognitoError(
                    category=category,
                    severity=severity,
                    message=f"Error in {func.__name__}: {str(e)}",
                    details=traceback.format_exc()
                )
                log_error(error, {'function': func.__name__, 'args': str(args)[:200]})
                
                if reraise:
                    raise
                return fallback_value
        return wrapper
    return decorator


def validate_input(
    value: Any,
    validation_func: Callable[[Any], bool],
    error_message: str,
    suggestions: Optional[str] = None
) -> Any:
    """Validate input with proper error handling."""
    try:
        if not validation_func(value):
            raise SecurityError(
                error_message,
                suggestions=suggestions,
                error_code="VALIDATION_FAILED"
            )
        return value
    except Exception as e:
        if isinstance(e, CognitoException):
            raise
        raise SecurityError(
            f"Validation error: {str(e)}",
            details=str(e),
            suggestions=suggestions
        )


def handle_file_size(file_size: int, max_size_mb: int = 10) -> None:
    """Validate file size."""
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise FileProcessingError(
            f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB)",
            suggestions=f"Please use a file smaller than {max_size_mb} MB",
            error_code="FILE_TOO_LARGE"
        )


def handle_code_length(code: str, max_length: int = 100000) -> None:
    """Validate code length."""
    if len(code) > max_length:
        raise SecurityError(
            f"Code length ({len(code)} characters) exceeds maximum allowed length ({max_length} characters)",
            suggestions=f"Please analyze code with less than {max_length} characters",
            error_code="CODE_TOO_LONG"
        )


def handle_api_rate_limit(requests_made: int, limit: int, window_minutes: int = 60) -> None:
    """Handle API rate limiting."""
    if requests_made >= limit:
        raise LLMAPIError(
            f"API rate limit exceeded: {requests_made}/{limit} requests in {window_minutes} minutes",
            suggestions=f"Please wait before making more requests or upgrade your plan",
            error_code="RATE_LIMIT_EXCEEDED"
        )


class PerformanceMonitor:
    """Monitor performance and detect issues."""
    
    def __init__(self):
        self.operation_times = {}
        self.slow_operations = []
        self.slow_threshold_seconds = 5.0
    
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self._record_operation_time(operation_name, execution_time)
                    
                    if execution_time > self.slow_threshold_seconds:
                        self._handle_slow_operation(operation_name, execution_time)
            return wrapper
        return decorator
    
    def _record_operation_time(self, operation_name: str, execution_time: float):
        """Record operation execution time."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        
        self.operation_times[operation_name].append(execution_time)
        
        # Keep only recent times (last 100)
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name].pop(0)
    
    def _handle_slow_operation(self, operation_name: str, execution_time: float):
        """Handle slow operations."""
        slow_op = {
            'operation': operation_name,
            'time': execution_time,
            'timestamp': time.time()
        }
        self.slow_operations.append(slow_op)
        
        # Keep only recent slow operations
        if len(self.slow_operations) > 50:
            self.slow_operations.pop(0)
        
        # Log performance warning
        log_error(
            CognitoError(
                category=ErrorCategory.PERFORMANCE,
                severity=ErrorSeverity.MEDIUM,
                message=f"Slow operation detected: {operation_name} took {execution_time:.2f} seconds",
                error_code="SLOW_OPERATION"
            )
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        
        return {
            'operation_stats': stats,
            'slow_operations': len(self.slow_operations),
            'recent_slow_operations': self.slow_operations[-10:]
        }


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def monitor_performance(operation_name: str):
    """Decorator for performance monitoring."""
    return get_performance_monitor().monitor_operation(operation_name)
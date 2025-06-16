"""
Configuration management for Cognito.
Handles environment variables, settings, and deployment configurations.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    feedback_db_path: str = "data/feedback.json"
    models_dir: str = "src/models"
    cache_dir: str = "data/cache"


@dataclass
class LLMConfig:
    """LLM integration configuration."""
    openai_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout: int = 30
    rate_limit_requests_per_minute: int = 60


@dataclass
class AnalysisConfig:
    """Code analysis configuration."""
    max_file_size_mb: int = 10
    max_lines_to_analyze: int = 5000
    supported_languages: list = field(default_factory=lambda: [
        'python', 'c', 'cpp', 'javascript', 'java', 'go', 'rust', 'php', 'ruby', 'csharp'
    ])
    enable_ml_readability: bool = True
    enable_llm_enhancement: bool = False
    enable_code_correction: bool = True
    enable_feedback_learning: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_input_validation: bool = True
    max_code_length: int = 100000  # 100KB
    allowed_file_extensions: list = field(default_factory=lambda: [
        '.py', '.c', '.cpp', '.h', '.js', '.java', '.go', '.rs', '.php', '.rb', '.cs'
    ])
    rate_limit_enabled: bool = True
    rate_limit_per_hour: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/cognito.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance and monitoring configuration."""
    enable_metrics: bool = True
    metrics_export_interval: int = 60  # seconds
    enable_profiling: bool = False
    cache_enabled: bool = True
    cache_ttl_hours: int = 24


@dataclass
class CognitoConfig:
    """Main Cognito configuration."""
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = True
    version: str = "0.8.0"
    
    # Component configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    def __post_init__(self):
        """Initialize configuration after object creation."""
        self._load_from_environment()
        self._create_directories()
        self._setup_logging()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Environment
        self.environment = os.getenv("COGNITO_ENV", self.environment)
        self.debug = os.getenv("COGNITO_DEBUG", str(self.debug)).lower() == "true"
        
        # API Keys
        self.llm.openai_api_key = os.getenv("OPENAI_API_KEY", self.llm.openai_api_key)
        self.llm.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", self.llm.huggingface_token)
        
        # Paths
        self.database.feedback_db_path = os.getenv("COGNITO_FEEDBACK_DB", self.database.feedback_db_path)
        self.database.models_dir = os.getenv("COGNITO_MODELS_DIR", self.database.models_dir)
        self.database.cache_dir = os.getenv("COGNITO_CACHE_DIR", self.database.cache_dir)
        
        # Analysis settings
        self.analysis.max_file_size_mb = int(os.getenv("COGNITO_MAX_FILE_SIZE_MB", self.analysis.max_file_size_mb))
        self.analysis.enable_llm_enhancement = os.getenv("COGNITO_ENABLE_LLM", str(self.analysis.enable_llm_enhancement)).lower() == "true"
        
        # Security
        self.security.rate_limit_per_hour = int(os.getenv("COGNITO_RATE_LIMIT", self.security.rate_limit_per_hour))
        
        # Logging
        self.logging.level = os.getenv("COGNITO_LOG_LEVEL", self.logging.level)
        self.logging.file_path = os.getenv("COGNITO_LOG_FILE", self.logging.file_path)
        
        # Production overrides
        if self.environment == "production":
            self.debug = False
            self.logging.level = "WARNING"
            self.security.rate_limit_enabled = True
            self.performance.enable_profiling = False
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.database.cache_dir,
            os.path.dirname(self.database.feedback_db_path),
            os.path.dirname(self.logging.file_path),
            self.database.models_dir
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.logging.file_enabled:
            from logging.handlers import RotatingFileHandler
            
            # Create file handler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            
            # Configure root logger
            logging.basicConfig(
                level=getattr(logging, self.logging.level.upper()),
                format=self.logging.format,
                handlers=[
                    logging.StreamHandler(),  # Console
                    file_handler  # File
                ]
            )
        else:
            logging.basicConfig(
                level=getattr(logging.logging.level.upper()),
                format=self.logging.format
            )
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_llm_enabled(self) -> bool:
        """Check if LLM features are available and enabled."""
        return (
            self.analysis.enable_llm_enhancement and 
            self.llm.openai_api_key is not None
        )
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required API keys for LLM features
        if self.analysis.enable_llm_enhancement and not self.llm.openai_api_key:
            issues.append("LLM enhancement enabled but OPENAI_API_KEY not provided")
        
        # Check file size limits
        if self.analysis.max_file_size_mb <= 0:
            issues.append("max_file_size_mb must be positive")
        
        if self.security.max_code_length <= 0:
            issues.append("max_code_length must be positive")
        
        # Check rate limiting
        if self.security.rate_limit_enabled and self.security.rate_limit_per_hour <= 0:
            issues.append("rate_limit_per_hour must be positive when rate limiting is enabled")
        
        # Check paths exist
        if not Path(self.database.models_dir).exists():
            issues.append(f"Models directory does not exist: {self.database.models_dir}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        
        return dataclass_to_dict(self)
    
    def get_safe_dict(self) -> Dict[str, Any]:
        """Get configuration dict with sensitive information removed."""
        config_dict = self.to_dict()
        
        # Remove sensitive information
        if 'llm' in config_dict:
            config_dict['llm'].pop('openai_api_key', None)
            config_dict['llm'].pop('huggingface_token', None)
        
        return config_dict


# Global configuration instance
_config: Optional[CognitoConfig] = None


def get_config() -> CognitoConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CognitoConfig()
    return _config


def reload_config() -> CognitoConfig:
    """Reload configuration from environment."""
    global _config
    _config = CognitoConfig()
    return _config


def validate_config() -> None:
    """Validate current configuration and raise error if invalid."""
    config = get_config()
    issues = config.validate()
    
    if issues:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))


# Convenience functions for common settings
def is_production() -> bool:
    """Check if running in production."""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development."""
    return get_config().is_development()


def get_llm_enabled() -> bool:
    """Check if LLM features are enabled."""
    return get_config().get_llm_enabled()


def get_models_dir() -> str:
    """Get models directory path."""
    return get_config().database.models_dir


def get_cache_dir() -> str:
    """Get cache directory path."""
    return get_config().database.cache_dir
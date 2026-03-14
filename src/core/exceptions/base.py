"""Base exception classes for the power forecasting system."""

from typing import Optional


class PowerForecastingError(Exception):
    """Base exception for power forecasting system."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        """Initialize base exception.
        
        Args:
            message: Error message
            error_code: Optional error code
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataError(PowerForecastingError):
    """Exception raised for data-related errors."""
    
    def __init__(self, message: str, error_code: str = "DATA_ERROR"):
        super().__init__(message, error_code)


class ModelError(PowerForecastingError):
    """Exception raised for model-related errors."""
    
    def __init__(self, message: str, error_code: str = "MODEL_ERROR"):
        super().__init__(message, error_code)


class ConfigError(PowerForecastingError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, error_code: str = "CONFIG_ERROR"):
        super().__init__(message, error_code)


class ValidationError(PowerForecastingError):
    """Exception raised for validation-related errors."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(message, error_code)


class DatabaseError(PowerForecastingError):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str, error_code: str = "DB_ERROR"):
        super().__init__(message, error_code)


class APIServiceError(PowerForecastingError):
    """Exception raised for API service errors."""
    
    def __init__(self, message: str, error_code: str = "API_ERROR"):
        super().__init__(message, error_code)


class ServiceError(PowerForecastingError):
    """Exception raised for service layer errors."""
    
    def __init__(self, message: str, error_code: str = "SERVICE_ERROR"):
        super().__init__(message, error_code)


class PipelineError(PowerForecastingError):
    """Exception raised for pipeline execution errors."""
    
    def __init__(self, message: str, error_code: str = "PIPELINE_ERROR"):
        super().__init__(message, error_code)


def handle_exception(exception: Exception, default_message: str = "An error occurred") -> PowerForecastingError:
    """Handle exception and convert to appropriate PowerForecastingError.
    
    Args:
        exception: Original exception
        default_message: Default message if conversion not possible
        
    Returns:
        PowerForecastingError instance
    """
    if isinstance(exception, PowerForecastingError):
        return exception
    
    # Convert common exceptions to appropriate PowerForecastingError types
    if isinstance(exception, (ValueError, TypeError)):
        return ValidationError(str(exception))
    elif isinstance(exception, FileNotFoundError):
        return DataError(f"File not found: {str(exception)}")
    elif isinstance(exception, KeyError):
        return ConfigError(f"Configuration key not found: {str(exception)}")
    else:
        return PowerForecastingError(f"{default_message}: {str(exception)}")
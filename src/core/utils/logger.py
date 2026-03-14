"""Logging utilities for the power forecasting system."""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


def setup_logger(
    name: str = "power_forecasting",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Optional format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "power_forecasting") -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Global logger instance
logger = setup_logger(
    name="power_forecasting",
    level=logging.DEBUG if os.getenv("DEBUG", "False").lower() == "true" else logging.INFO,
    log_file="logs/app.log"
)


def log_execution(func):
    """Decorator to log function execution."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


# Export common logging levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
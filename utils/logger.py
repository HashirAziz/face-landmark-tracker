"""
Logging configuration using loguru for better console output.
"""

import sys
from loguru import logger
from config.settings import Config


def setup_logger():
    """
    Configure logger with custom format and level.
    
    Returns:
        logger: Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=Config.LOG_LEVEL,
        colorize=True
    )
    
    return logger


# Create global logger instance
log = setup_logger()
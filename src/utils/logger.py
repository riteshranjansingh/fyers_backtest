"""
Logging configuration for the entire project
"""
import os
import logging
from datetime import datetime

def setup_logger(name: str = None, log_file: str = None, level=logging.INFO):
    """
    Setup a logger with console and file handlers
    
    Args:
        name: Logger name (defaults to root logger)
        log_file: Optional log file name
        level: Logging level
        
    Returns:
        logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create project-wide loggers
app_logger = setup_logger('app', 'app.log')
trading_logger = setup_logger('trading', 'trading.log')
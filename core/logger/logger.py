# core/logger/logger.py

import os
import logging
import logging.config
import yaml
from pathlib import Path

from core.utils import get_project_root

class LoggerManager:
    """Centralized logger management for the core module."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LoggerManager._initialized:
            self.setup_logging()
            LoggerManager._initialized = True
    
    def setup_logging(
        self,
        default_path='core/configs/logging.yaml',
        default_level=logging.WARNING,
        env_key='LOG_CFG'
    ):
        """Setup logging configuration

        Args:
            default_path (str): Path to the logging configuration file
            default_level (int): Default logging level if config fails
            env_key (str): Environment variable that can override config path
        """
        # Create logs directory if it doesn't exist
        log_dir = get_project_root() / 'core/logger/logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        path = os.getenv(env_key, default_path)
        
        if os.path.exists(path):
            with open(path, 'rt') as f:
                try:
                    config = yaml.safe_load(f.read())
                    logging.config.dictConfig(config)
                    logging.info(f"Logging configuration loaded from {path}")
                except Exception as e:
                    print(f'Error in Logging Configuration: {e}')
                    print('Using default logging configuration')
                    self._setup_default_logging(default_level)
        else:
            print(f'Configuration file not found at {path}')
            print('Using default logging configuration')
            self._setup_default_logging(default_level)
    
    def _setup_default_logging(self, default_level):
        """Setup basic default logging if configuration fails."""
        log_dir = get_project_root() / 'core/logger/logs'
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s | %(pathname)s:%(lineno)d | %(levelname)s | %(name)s | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(log_dir / 'default.log'))
            ]
        )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name (str): Name of the logger, typically __name__ of the module
        
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    # Ensure logger is initialized
    LoggerManager()
    return logging.getLogger(name)

# Initialize logging when the module is imported
LoggerManager() 
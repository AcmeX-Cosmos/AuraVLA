"""
ROS Logger

Structured logging utilities for AuraVLA system.
"""

import logging
from typing import Optional


class EVALogger:
    """
    Structured logger for AuraVLA system
    """

    def __init__(self, name: str, level: str = 'INFO'):
        """
        Initialize logger

        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message"""
        self.logger.critical(msg)

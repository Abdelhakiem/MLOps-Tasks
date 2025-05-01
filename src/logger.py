from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

from loguru import logger


class ExecutorLogger:
    """
    A logger class that utilizes the Loguru library to provide a flexible and
    customizable logging interface. It supports logging to both console and
    file with different formats and log levels.

    Attributes:
        logger: An instance of the Loguru logger.
    """

    def __init__(self, logs_path: Optional[str] = None, level: str = "INFO"):
        """
        Initializes the ExecutorLogger.

        Args:
            logs_path (Optional[str]): The path of the log files to which logs
                                       will be written. If None, logging to file
                                       is disabled.
            level (str): The logging level (e.g., "DEBUG", "INFO", "WARNING",
                          "ERROR", "CRITICAL"). Default is "INFO".
        """
        self.logger = logger
        self.logger.remove()  # Remove the default logger
        self.logger.add(sys.stdout, level=level, format=self._get_console_format())

        if logs_path:
            self.logger.add(
                Path("logs")
                / Path(logs_path)
                / f"logs_{datetime.now().strftime('%Y%m%d')}.log",
                level=level,
                format=self._get_file_format(),
                rotation="10 MB",
            )

    @staticmethod
    def _get_console_format() -> str:
        """
        Provides the format for console logging.

        Returns:
            str: The format string for console log messages.
        """
        return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

    @staticmethod
    def _get_file_format() -> str:
        """
        Provides the format for file logging.

        Returns:
            str: The format string for file log messages.
        """
        return "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"

    def debug(self, message: str, *args, **kwargs):
        """
        Logs a message with level DEBUG.

        Args:
            message (str): The message to log.
            *args: Additional positional arguments to format the message.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """
        Logs a message with level INFO.

        Args:
            message (str): The message to log.
            *args: Additional positional arguments to format the message.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """
        Logs a message with level WARNING.

        Args:
            message (str): The message to log.
            *args: Additional positional arguments to format the message.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """
        Logs a message with level ERROR.

        Args:
            message (str): The message to log.
            *args: Additional positional arguments to format the message.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """
        Logs a message with level CRITICAL.

        Args:
            message (str): The message to log.
            *args: Additional positional arguments to format the message.
            **kwargs: Additional keyword arguments for logging.
        """
        self.logger.critical(message, *args, **kwargs)
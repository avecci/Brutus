import os
import logging
from pythonjsonlogger import jsonlogger
from pythonjsonlogger.json import JsonFormatter


def setup_logger(name=None):
    """
    Set up a logger with JSON formatting.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(os.path.join("logs", "brutus.log"))
    stream_handler = logging.StreamHandler()

    # Create formatter
    file_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(timestamp)s %(levelname)s %(message)s ", timestamp=True
        )
    )
    stream_handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(timestamp)s %(levelname)s %(message)s ", timestamp=True
        )
    )

    # Add both handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

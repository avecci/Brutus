"""Custom logger to properly structure JSON formatted logs."""
import logging
import os
from datetime import datetime

from pythonjsonlogger import json


def setup_logger(name=None, log_folder="logs", log_filename="brutus"):
    """Set up a logger with JSON formatting.

    Args:
        name (str): Name of the logger
        log_folder (str): Log folder name
        log_filename (str): Log file name

    Returns:
        logging.Logger: Configured logger instance
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create log file path with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_folder, f"{log_filename}.{current_date}.log")

    # Use regular FileHandler since we're creating dated files directly
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    # Custom log record format
    custom_format = (
        "%(timestamp)s %(levelname)s %(name)s "
        "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )

    file_handler.setFormatter(json.JsonFormatter(custom_format, timestamp=True))
    stream_handler.setFormatter(json.JsonFormatter(custom_format, timestamp=True))

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    return logger

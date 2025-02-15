import os
import logging
from pythonjsonlogger import json
from pythonjsonlogger.json import JsonFormatter


def setup_logger(name=None, log_folder="logs", log_filename="brutus.log"):
    """
    Set up a logger with JSON formatting.

    Args:
        name (str): Name of the logger
        log_folder (str): Log folder name
        log_filename (str): Log file name

    Returns:
        logging.Logger: Configured logger instance
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_folder, log_filename))
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(
        json.JsonFormatter("%(timestamp)s %(levelname)s %(message)s ", timestamp=True)
    )
    stream_handler.setFormatter(
        json.JsonFormatter("%(timestamp)s %(levelname)s %(message)s ", timestamp=True)
    )

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    return logger

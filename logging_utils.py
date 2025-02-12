import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json

class CustomJSONFormatter(logging.Formatter):
    """
    JSON formatter that includes only specified fields in the log output.
    
    Attributes:
        ALLOWED_KEYS (set): Set of keys that will be included in the JSON output
    """
    ALLOWED_KEYS = {'level', 'msg', 'time', 'name', 'target_image', 'library_folder', 
                    'similarity_threshold', 'image_path', 'results', 'error', 'params'}

    def format(self, record):
        if hasattr(record, "__dict__"):
            record_dict = {
                'level': record.levelname,
                'msg': record.msg,
                'name': record.name
            }
            
            # Add any extra fields that are in ALLOWED_KEYS
            if hasattr(record, 'extra'):
                for key, value in record.__dict__.items():
                    if key in self.ALLOWED_KEYS:
                        record_dict[key] = value

            # Convert dictionary to JSON string
            return json.dumps(record_dict)
        return "{}"

def setup_logger(name=__name__, log_path="./logs/brutus.log"):
    """
    Configure and return a logger with both file and console handlers.
    
    Args:
        name (str): Logger name, defaults to module name
        log_path (str): Path where log files will be stored
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    Path(log_path).parent.mkdir(exist_ok=True)

    # Create formatters and filters
    formatter = CustomJSONFormatter()

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

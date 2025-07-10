import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, APP_NAME

def setup_logging():
    """
    Configures the logging for the application.
    """
    numeric_level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level: {LOG_LEVEL}. Defaulting to INFO.")
        numeric_level = logging.INFO

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate logs
    # This is important if this function might be called multiple times
    # or if other libraries (like uvicorn in FastAPI) configure logging.
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional)
    if LOG_FILE:
        try:
            file_handler = logging.FileHandler(LOG_FILE, mode='a') # 'a' for append
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logging.info(f"Logging to file: {LOG_FILE}")
        except Exception as e:
            logging.error(f"Failed to set up log file at {LOG_FILE}: {e}", exc_info=True)

    # To control logging from specific libraries (e.g., noisy libraries)
    # logging.getLogger("httpx").setLevel(logging.WARNING) # Example for httpx if used by openai
    # logging.getLogger("openai").setLevel(logging.WARNING) # Example for openai library

    logging.info(f"Logging initialized for {APP_NAME} with level {LOG_LEVEL}.")

if __name__ == "__main__":
    # This is for testing the logging setup
    # It requires config.py to be able to load LOG_LEVEL etc.
    # To run this directly, you might need to adjust Python's path or run as a module.
    # For simplicity, assuming config.py is accessible.

    # First, ensure config is loaded (as setup_logging depends on it)
    from .config import load_config
    load_config()

    # Now setup logging
    setup_logging()

    # Test logging
    root_logger = logging.getLogger()
    root_logger.debug("This is a debug message.")
    root_logger.info("This is an info message.")
    root_logger.warning("This is a warning message.")
    root_logger.error("This is an error message.")
    root_logger.critical("This is a critical message.")

    # Test logging from a specific module
    module_logger = logging.getLogger("my_module")
    module_logger.info("Info message from my_module.")

    if LOG_FILE:
        print(f"Check the log file: {LOG_FILE} (if configured and writable)")

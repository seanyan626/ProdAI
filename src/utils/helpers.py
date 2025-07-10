# src/utils/helpers.py
import logging
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file and returns its content as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Saves a dictionary to a JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")
        raise

# Example of another utility function
def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Truncates text to a maximum length, adding an ellipsis if truncated.
    """
    if len(text) > max_length:
        return text[:max_length - len(ellipsis)] + ellipsis
    return text

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    test_data = {"name": "Test Data", "version": 1, "items": [1, 2, 3]}
    test_file_path = "test_helpers.json"

    logger.info(f"Saving test data to {test_file_path}...")
    save_json_file(test_data, test_file_path)

    logger.info(f"Loading test data from {test_file_path}...")
    loaded_data = load_json_file(test_file_path)
    logger.info(f"Loaded data: {loaded_data}")

    assert test_data == loaded_data
    logger.info("JSON load/save test successful.")

    original_text = "This is a very long string that needs to be truncated for display purposes."
    truncated = truncate_text(original_text, 30)
    logger.info(f"Original: '{original_text}'")
    logger.info(f"Truncated: '{truncated}'")
    assert truncated == "This is a very long string..."

    # Clean up test file
    import os
    os.remove(test_file_path)
    logger.info(f"Cleaned up {test_file_path}.")

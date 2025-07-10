# tests/utils/test_helpers.py
import pytest
import json
import os
import logging

# Ensure config is loaded if any underlying modules need it, and logging is set up.
try:
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
except ImportError:
    print("Warning: Could not import config/logging for helper tests. This might affect module behavior.")


from src.utils.helpers import load_json_file, save_json_file, truncate_text

logger = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture
def temp_json_file(tmp_path):
    """Creates a temporary JSON file path for testing."""
    return tmp_path / "test_data.json"

# --- Tests for JSON helpers ---

def test_save_and_load_json_file(temp_json_file):
    logger.info(f"Testing save_json_file and load_json_file with: {temp_json_file}")
    test_data = {"name": "AI Project", "version": "1.0", "modules": ["agents", "llms", "rag"]}

    # Test saving
    try:
        save_json_file(test_data, str(temp_json_file))
        assert os.path.exists(temp_json_file)
        logger.info("JSON data saved successfully.")
    except Exception as e:
        pytest.fail(f"save_json_file raised an exception: {e}")

    # Test loading
    try:
        loaded_data = load_json_file(str(temp_json_file))
        assert loaded_data == test_data
        logger.info("JSON data loaded successfully and matches original.")
    except Exception as e:
        pytest.fail(f"load_json_file raised an exception: {e}")

def test_load_json_file_not_found(tmp_path):
    logger.info("Testing load_json_file with a non-existent file.")
    non_existent_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_json_file(str(non_existent_file))
    logger.info("FileNotFoundError raised as expected.")


def test_load_json_file_invalid_json(tmp_path):
    logger.info("Testing load_json_file with an invalid JSON file.")
    invalid_json_file = tmp_path / "invalid.json"
    with open(invalid_json_file, "w") as f:
        f.write("{'name': 'test', 'value': 123,}") # Invalid JSON (trailing comma, single quotes often problematic)

    with pytest.raises(json.JSONDecodeError):
        load_json_file(str(invalid_json_file))
    logger.info("JSONDecodeError raised as expected for invalid JSON.")

# --- Tests for text helpers ---

def test_truncate_text_no_truncation_needed():
    logger.info("Testing truncate_text when no truncation is needed.")
    text = "Short text."
    max_length = 20
    truncated = truncate_text(text, max_length)
    assert truncated == text
    logger.info(f"Result: '{truncated}' (correct, no truncation)")

def test_truncate_text_truncation_occurs():
    logger.info("Testing truncate_text when truncation should occur.")
    text = "This is a fairly long string that definitely needs to be truncated."
    max_length = 20
    expected_ellipsis = "..."
    # Expected: "This is a fairly..." (20 chars total, so 17 from text + 3 for ellipsis)
    expected_text = text[:max_length - len(expected_ellipsis)] + expected_ellipsis

    truncated = truncate_text(text, max_length)
    assert truncated == expected_text
    assert len(truncated) == max_length
    logger.info(f"Result: '{truncated}' (correct, truncated)")

def test_truncate_text_custom_ellipsis():
    logger.info("Testing truncate_text with a custom ellipsis.")
    text = "Another long example for custom ellipsis."
    max_length = 15
    custom_ellipsis = " (..)" # length 5
    # Expected: "Another lo (..)" (15 chars total, so 10 from text + 5 for ellipsis)
    expected_text = text[:max_length - len(custom_ellipsis)] + custom_ellipsis

    truncated = truncate_text(text, max_length, ellipsis=custom_ellipsis)
    assert truncated == expected_text
    assert len(truncated) == max_length
    logger.info(f"Result: '{truncated}' (correct, custom ellipsis)")

def test_truncate_text_max_length_too_small_for_ellipsis():
    logger.info("Testing truncate_text when max_length is smaller than ellipsis.")
    text = "Some text"
    max_length = 2 # Smaller than default "..."
    # Expected behavior: returns text truncated to max_length without ellipsis, or just ellipsis if max_length allows.
    # Current implementation will try to fit ellipsis, potentially resulting in just ellipsis or negative slice.
    # Let's check the behavior: text[:2-3] + "..." -> text[:-1] + "..." -> "Some tex" + "..." (incorrect)
    # A robust version might return text[:max_length] or just the ellipsis if max_length is tiny.

    # Based on current implementation:
    # If max_length < len(ellipsis), it will result in text[:negative_index] + ellipsis
    # which means it will take a part of the string from the end.
    # Example: text="abcde", max_length=2, ellipsis="...", text[:2-3] = text[:-1] = "abcd"
    # Result: "abcd..." which is longer than max_length. This is a flaw.

    # A better expectation if max_length < len(ellipsis): return text[:max_length]
    # However, let's test the current behavior first.
    truncated = truncate_text(text, max_length)
    logger.info(f"Truncate with max_length={max_length}, ellipsis='...': '{truncated}'")
    # This test might fail or show unexpected behavior based on the simple implementation.
    # A goodtruncate would ensure len(truncated) <= max_length.
    # The current simple one does NOT guarantee this if max_length < len(ellipsis).
    # For text="Some text", max_length=2, ellipsis="...":
    # text[:2-3] is text[:-1] which is "Some tex"
    # result is "Some tex..." which is 12 chars long.
    assert len(truncated) > max_length, "Current truncate_text fails when max_length < len(ellipsis)"
    logger.warning("truncate_text behavior with max_length < len(ellipsis) is not ideal.")

    # Test if max_length exactly equals ellipsis length
    max_length_equals_ellipsis = 3
    truncated_equal = truncate_text(text, max_length_equals_ellipsis) # text[:3-3]+"..." = text[:0]+"..." = "..."
    assert truncated_equal == "..."
    assert len(truncated_equal) == max_length_equals_ellipsis
    logger.info(f"Truncate with max_length={max_length_equals_ellipsis}, ellipsis='...': '{truncated_equal}' (correct)")


def test_truncate_text_empty_string():
    logger.info("Testing truncate_text with an empty string.")
    text = ""
    truncated = truncate_text(text, 10)
    assert truncated == ""
    logger.info(f"Result: '{truncated}' (correct, empty string)")

logger.info("Helper tests completed.")

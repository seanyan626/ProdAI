# src/prompts/prompt_manager.py
import logging
import os
from typing import Dict, Any, Optional, List
from string import Template # Using string.Template for simple variable substitution

# Assuming your project structure is aiproject/src/prompts/templates
# and this file is aiproject/src/prompts/prompt_manager.py
DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

logger = logging.getLogger(__name__)

class PromptTemplate(Template):
    """
    Custom Template class to allow for different delimiter if needed,
    and to potentially add more complex template logic in the future.
    For now, it's a direct subclass of string.Template.
    """
    # delimiter = "$" # Default delimiter, can be changed if necessary
    pass


class PromptManager:
    """
    Manages loading, formatting, and accessing prompt templates.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initializes the PromptManager.

        Args:
            templates_dir (Optional[str]): The directory where prompt template files are stored.
                                           Defaults to "templates" subdirectory next to this file.
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.loaded_templates: Dict[str, PromptTemplate] = {}
        self._load_all_templates()
        logger.info(f"PromptManager initialized. Loaded templates from: {self.templates_dir}")

    def _load_template_file(self, file_path: str) -> Optional[str]:
        """Loads a single template file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            logger.error(f"Template file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading template file {file_path}: {e}", exc_info=True)
            return None

    def _load_all_templates(self) -> None:
        """
        Loads all .txt or .prompt files from the templates directory.
        The filename (without extension) is used as the template name.
        """
        if not os.path.isdir(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}. No templates loaded.")
            return

        for filename in os.listdir(self.templates_dir):
            if filename.endswith((".txt", ".prompt")):
                template_name = os.path.splitext(filename)[0]
                file_path = os.path.join(self.templates_dir, filename)
                content = self._load_template_file(file_path)
                if content:
                    self.loaded_templates[template_name] = PromptTemplate(content)
                    logger.debug(f"Loaded template: '{template_name}' from {filename}")
        logger.info(f"Loaded {len(self.loaded_templates)} templates.")


    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        Retrieves a loaded prompt template.

        Args:
            template_name (str): The name of the template (filename without extension).

        Returns:
            Optional[PromptTemplate]: The PromptTemplate object, or None if not found.
        """
        template = self.loaded_templates.get(template_name)
        if not template:
            logger.warning(f"Template '{template_name}' not found.")
        return template

    def format_prompt(self, template_name: str, **kwargs: Any) -> Optional[str]:
        """
        Formats a prompt using a named template and provided keyword arguments.

        Args:
            template_name (str): The name of the template to use.
            **kwargs: The variables to substitute into the template.

        Returns:
            Optional[str]: The formatted prompt string, or None if the template is not found
                           or if there's an error during formatting.
        """
        template = self.get_template(template_name)
        if not template:
            return None

        try:
            # Using safe_substitute to avoid KeyError if some placeholders are missing
            # and to allow for $ signs not intended for substitution.
            # If strict substitution is needed, use substitute().
            formatted_prompt = template.safe_substitute(**kwargs)
            logger.debug(f"Formatted prompt '{template_name}' with variables: {kwargs}")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"Missing variable for template '{template_name}': {e}. Variables provided: {kwargs}")
            return None
        except Exception as e:
            logger.error(f"Error formatting template '{template_name}': {e}", exc_info=True)
            return None

    def list_available_templates(self) -> List[str]:
        """
        Returns a list of names of all loaded templates.
        """
        return list(self.loaded_templates.keys())

# --- Example: Create a dummy template file for testing ---
def _create_dummy_template_files(templates_dir: str):
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # Template 1: simple_greeting.txt
    with open(os.path.join(templates_dir, "simple_greeting.txt"), "w", encoding="utf-8") as f:
        f.write("Hello, $name! Welcome to $platform.")

    # Template 2: question_template.prompt
    with open(os.path.join(templates_dir, "question_template.prompt"), "w", encoding="utf-8") as f:
        f.write("User query: $query\nContext: $context\n\nPlease answer the query based on the context.")

    # Template 3: with_missing_var_test.txt (for safe_substitute testing)
    with open(os.path.join(templates_dir, "escape_test.txt"), "w", encoding="utf-8") as f:
        f.write("This template costs $$10. Your variable is $var.")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    logger.info("Testing PromptManager...")

    # Create dummy templates for the test
    test_templates_path = os.path.join(os.path.dirname(__file__), "test_prompt_templates")
    _create_dummy_template_files(test_templates_path)

    manager = PromptManager(templates_dir=test_templates_path)

    logger.info(f"Available templates: {manager.list_available_templates()}")
    assert "simple_greeting" in manager.list_available_templates()
    assert "question_template" in manager.list_available_templates()

    # Test formatting a valid prompt
    greeting = manager.format_prompt("simple_greeting", name="Alice", platform="AI World")
    logger.info(f"Formatted 'simple_greeting': {greeting}")
    assert greeting == "Hello, Alice! Welcome to AI World."

    # Test formatting another valid prompt
    question = manager.format_prompt(
        "question_template",
        query="What is the capital of France?",
        context="France is a country in Europe. Its capital is Paris."
    )
    logger.info(f"Formatted 'question_template': {question}")
    assert "User query: What is the capital of France?" in question
    assert "Context: France is a country in Europe. Its capital is Paris." in question

    # Test formatting with missing variables (using safe_substitute)
    escaped_prompt = manager.format_prompt("escape_test", var="test_value")
    logger.info(f"Formatted 'escape_test': {escaped_prompt}")
    assert escaped_prompt == "This template costs $10. Your variable is test_value."

    # Test formatting with all variables missing for a template
    missing_vars_prompt = manager.format_prompt("simple_greeting")
    logger.info(f"Formatted 'simple_greeting' with no vars: {missing_vars_prompt}")
    assert missing_vars_prompt == "Hello, $name! Welcome to $platform."


    # Test getting a non-existent template
    non_existent = manager.format_prompt("non_existent_template", key="value")
    assert non_existent is None

    # Clean up dummy templates
    import shutil
    try:
        shutil.rmtree(test_templates_path)
        logger.info(f"Cleaned up test templates directory: {test_templates_path}")
    except Exception as e:
        logger.error(f"Error cleaning up test templates: {e}")

    logger.info("PromptManager tests completed successfully.")

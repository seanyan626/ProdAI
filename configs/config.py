import os
import logging
from dotenv import load_dotenv

# --- Application Metadata ---
APP_NAME = "AI Project Framework"
APP_VERSION = "0.1.0"


# --- Environment Variable Loading ---
def load_env():
    """
    Load environment variables from .env file.
    Looks for .env in the current directory or parent directories.
    """
    # Determine the base directory (project root)
    # This assumes config.py is in a subdirectory of the project root.
    # Adjust as necessary if your structure is different.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Moves up one level from configs to project root

    dotenv_path = os.path.join(project_root, ".env")

    if not os.path.exists(dotenv_path):
        logging.warning(
            f".env file not found at {dotenv_path}. "
            "Please ensure it exists and contains necessary configurations. "
            "You can copy .env.example to .env and fill in your details."
        )
        # Fallback to trying to load from current working directory if needed,
        # though explicit path is better.
        load_dotenv()
    else:
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded environment variables from {dotenv_path}")


_ENV_LOADED = False

def load_config():
    """
    Loads all configurations. Ensures .env is loaded only once.
    """
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_env()
        _ENV_LOADED = True

# --- API Keys and Endpoints ---
# Load them after calling load_config()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# --- LLM Settings ---
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1500"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# --- Agent Settings ---
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))


# --- RAG Settings ---
# Example: Path to knowledge base, can be relative to project root
# KNOWLEDGE_BASE_PATH = os.path.join(os.getenv("PROJECT_ROOT_DIR", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "knowledge_base")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
# VECTOR_STORE_PATH = os.path.join(os.getenv("PROJECT_ROOT_DIR", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "vector_store")


# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", None) # e.g., "app.log"


# --- Other Configurations ---
# Add any other global configurations your application might need.


# --- Helper function to ensure config is loaded before accessing variables ---
# This is useful if modules import specific config variables directly.
# Call load_config() at the beginning of your main application script (e.g., main.py)
# or ensure it's called before any config variable is accessed.

# Example of how to access a config value:
# from configs.config import OPENAI_API_KEY, load_config
# load_config() # Ensure it's loaded
# print(OPENAI_API_KEY)

if __name__ == "__main__":
    # This part is for testing the config loading
    load_config()
    print(f"App Name: {APP_NAME}")
    print(f"OpenAI API Key: {'Loaded' if OPENAI_API_KEY else 'Not Found'}")
    print(f"Default LLM Model: {DEFAULT_LLM_MODEL}")
    print(f"Log Level: {LOG_LEVEL}")
    # print(f"Knowledge Base Path: {KNOWLEDGE_BASE_PATH}") # Uncomment if using
    # project_root_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f"Project root (derived): {project_root_test}")
    # print(f"Full path to .env being checked: {os.path.join(project_root_test, '.env')}")

## Agent Instructions for `aiproject`

This file provides special instructions for AI agents working with this codebase.

### Project Overview

`aiproject` is a Python-based framework for building AI applications leveraging Large Language Models (LLMs). It features a modular architecture with components for agents, LLM interaction, memory, prompt management, RAG (Retrieval Augmented Generation), and tools.

### Key Architectural Principles

*   **Modularity**: Each core functionality (agents, llms, memory, etc.) resides in its own directory under `src/`. Aim to keep these modules self-contained.
*   **High Cohesion, Low Coupling**: Classes and functions within a module should be closely related. Dependencies between modules should be minimized and managed through well-defined interfaces (e.g., base classes).
*   **Configuration Management**: All configurations, including API keys and model parameters, should be managed via environment variables (loaded from a `.env` file, based on `.env.example`) and accessed through the `configs` module. Do not hardcode sensitive information or configurable parameters directly in the code.
*   **Extensibility**: The system is designed to be extensible. When adding new LLMs, agent types, memory systems, or tools, prefer creating new classes that inherit from the provided base classes in each module.

### Working with Modules

*   **`src/agents`**:
    *   New agent types should inherit from `BaseAgent`.
    *   Agents should primarily interact with other system components (LLMs, tools, memory) through their respective abstractions.
*   **`src/llms`**:
    *   New LLM integrations should inherit from `BaseLLM`.
    *   Focus on abstracting the specific API calls of the LLM provider.
*   **`src/memory`**:
    *   New memory systems should inherit from `BaseMemory`.
*   **`src/prompts`**:
    *   Use the `PromptManager` for loading and formatting prompts.
    *   Store prompt templates in the `src/prompts/templates/` directory.
*   **`src/rag`**:
    *   Components like retrievers and document loaders should be designed to be swappable.
*   **`src/tools`**:
    *   New tools should inherit from `BaseTool`.
    *   Tools should be self-contained and have a clear `execute` method.

### Testing

*   All new features or bug fixes should be accompanied by relevant unit tests in the `tests/` directory.
*   Maintain a similar directory structure in `tests/` as in `src/`.
*   Run tests frequently to ensure code quality. (Command to run tests will be specified once a test runner is set up, e.g., `pytest`).

### Code Style and Conventions

*   Follow PEP 8 Python style guidelines.
*   Use type hinting for all function signatures and variable declarations.
*   Write clear and concise docstrings for all modules, classes, and functions.

### Committing Changes

*   Write clear and descriptive commit messages.
*   Ensure all tests pass before committing.

### Environment Setup

1.  Copy `.env.example` to `.env`.
2.  Fill in the necessary API keys and configurations in the `.env` file.

By following these guidelines, you will help maintain the quality, consistency, and maintainability of the `aiproject` codebase.

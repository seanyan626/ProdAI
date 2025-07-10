# src/memory/base_memory.py
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union

class BaseMemory(ABC):
    """
    Abstract base class for memory systems used by agents.
    """

    @abstractmethod
    def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
        """
        Adds a message or interaction to the memory.

        Args:
            message (Union[str, Dict[str, Any]]): The message content.
                Can be a simple string or a structured dictionary (e.g., OpenAI message format).
            role (Optional[str]): The role associated with the message (e.g., "user", "assistant", "system").
                                   Required if message is a simple string.
        """
        pass

    @abstractmethod
    def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history.

        Args:
            max_messages (Optional[int]): The maximum number of recent messages to retrieve.
            max_tokens (Optional[int]): The maximum number of tokens the history should roughly correspond to.
                                        (Implementation might involve token counting and truncation).

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries, typically in a format
                                  compatible with LLM chat APIs (e.g., [{"role": "user", "content": "..."}, ...]).
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clears the entire memory.
        """
        pass

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the most recent message from memory.

        Returns:
            Optional[Dict[str, Any]]: The last message, or None if memory is empty.
        """
        history = self.get_history(max_messages=1)
        return history[-1] if history else None

    # Optional: Methods for more complex memory operations like summarization or pruning
    # def summarize(self) -> str:
    #     """ Returns a summary of the conversation. """
    #     raise NotImplementedError
    #
    # def prune(self, strategy: str = "fifo", max_size: Optional[int] = None) -> None:
    #     """ Prunes the memory based on a strategy. """
    #     raise NotImplementedError

if __name__ == '__main__':
    # Example of how a concrete implementation might be tested (conceptual)
    class SimpleTestMemory(BaseMemory):
        def __init__(self):
            self.history: List[Dict[str, Any]] = []

        def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
            if isinstance(message, str):
                if not role:
                    raise ValueError("Role must be provided if message is a string.")
                self.history.append({"role": role, "content": message})
            elif isinstance(message, dict) and "content" in message and "role" in message:
                self.history.append(message)
            else:
                raise TypeError("Message must be a string (with role) or a dict with 'role' and 'content'.")
            print(f"Memory: Added message - Role: {self.history[-1]['role']}, Content: '{self.history[-1]['content'][:50]}...'")

        def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
            # Basic max_messages, token counting not implemented for this simple test
            if max_messages is not None and max_messages > 0:
                return self.history[-max_messages:]
            return list(self.history) # Return a copy

        def clear(self) -> None:
            self.history = []
            print("Memory: Cleared.")

    # Test the SimpleTestMemory
    print("Testing BaseMemory with SimpleTestMemory implementation...")
    memory = SimpleTestMemory()
    memory.add_message("Hello there!", role="user")
    memory.add_message({"role": "assistant", "content": "Hi! How can I help you today?"})
    memory.add_message("Tell me a joke.", role="user")

    print("\nFull History:")
    for msg in memory.get_history():
        print(msg)

    print("\nLast 2 Messages:")
    for msg in memory.get_history(max_messages=2):
        print(msg)

    print(f"\nLast Message: {memory.get_last_message()}")

    memory.clear()
    print(f"\nHistory after clear: {memory.get_history()}")
    assert not memory.get_history()
    print("BaseMemory conceptual test finished.")

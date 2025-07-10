# src/memory/simple_memory.py
import logging
from typing import Any, List, Dict, Optional, Union

from .base_memory import BaseMemory

logger = logging.getLogger(__name__)

class SimpleMemory(BaseMemory):
    """
    A simple in-memory store for conversation history.
    It stores messages as a list of dictionaries.
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        Initializes the simple memory.

        Args:
            system_message (Optional[str]): An optional system message to prepend to the history.
        """
        self.history: List[Dict[str, Any]] = []
        self.system_message_content: Optional[str] = system_message
        if system_message:
            # Add system message if provided, but don't log it as a regular "add"
            self._prepend_system_message()
        logger.info(f"SimpleMemory initialized. System message: {'Set' if system_message else 'Not set'}")

    def _prepend_system_message(self):
        """Internal method to add or ensure the system message is at the start."""
        if not self.system_message_content:
            return

        # Remove existing system message if it's there to avoid duplicates on re-init or modification
        self.history = [msg for msg in self.history if msg.get("role") != "system"]
        self.history.insert(0, {"role": "system", "content": self.system_message_content})


    def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
        """
        Adds a message to the memory.

        Args:
            message (Union[str, Dict[str, Any]]): The message content.
                If string, `role` must be provided.
                If dict, must contain "role" and "content" keys.
            role (Optional[str]): The role of the message sender (e.g., "user", "assistant").
                                   Ignored if message is a dict.
        """
        if isinstance(message, str):
            if not role:
                logger.error("Role must be provided when message is a string.")
                raise ValueError("Role must be provided when message is a string.")
            msg_dict = {"role": role, "content": message}
        elif isinstance(message, dict):
            if "role" not in message or "content" not in message:
                logger.error("Message dictionary must contain 'role' and 'content' keys.")
                raise ValueError("Message dictionary must contain 'role' and 'content' keys.")
            msg_dict = message
        else:
            logger.error(f"Unsupported message type: {type(message)}")
            raise TypeError("Message must be a string or a dictionary.")

        self.history.append(msg_dict)
        logger.debug(f"Added to SimpleMemory: Role: {msg_dict['role']}, Content: '{str(msg_dict['content'])[:70]}...'")

    def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history.

        Args:
            max_messages (Optional[int]): The maximum number of recent messages to retrieve.
                                        If a system message exists, it is always included and
                                        not counted towards max_messages.
            max_tokens (Optional[int]): (Not implemented in SimpleMemory) Maximum tokens for history.

        Returns:
            List[Dict[str, Any]]: The list of message dictionaries.
        """
        if max_tokens is not None:
            logger.warning("max_tokens is not implemented in SimpleMemory. Returning based on max_messages.")

        # Ensure system message is preserved if it exists
        current_history = list(self.history) # Work with a copy
        output_history: List[Dict[str, Any]] = []

        system_msg = None
        if current_history and current_history[0].get("role") == "system":
            system_msg = current_history.pop(0) # Remove system message for now

        if max_messages is not None and max_messages > 0:
            # Get the last 'max_messages' from the non-system part of history
            output_history = current_history[-max_messages:]
        else:
            output_history = current_history # Get all non-system messages

        if system_msg:
            # Add system message back to the beginning
            output_history.insert(0, system_msg)

        logger.debug(f"Retrieved {len(output_history)} messages from SimpleMemory.")
        return output_history

    def clear(self) -> None:
        """
        Clears all messages from the memory, except for the initial system message if one was set.
        """
        self.history = []
        if self.system_message_content:
            self._prepend_system_message() # Re-add system message after clearing
        logger.info("SimpleMemory cleared (retained system message if configured).")

    def set_system_message(self, system_message: str) -> None:
        """
        Sets or updates the system message.
        """
        self.system_message_content = system_message
        self._prepend_system_message()
        logger.info(f"System message updated in SimpleMemory: '{system_message[:70]}...'")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config() # Ensure environment variables are loaded for any underlying components
    setup_logging() # Configure logging for the test

    logger.info("Testing SimpleMemory...")

    # Test without system message
    memory1 = SimpleMemory()
    memory1.add_message("Hello", role="user")
    memory1.add_message({"role": "assistant", "content": "Hi there!"})
    history1 = memory1.get_history()
    logger.info(f"Memory1 History (no system message): {history1}")
    assert len(history1) == 2
    assert history1[0]["content"] == "Hello"

    # Test with system message
    sys_msg = "You are a helpful AI."
    memory2 = SimpleMemory(system_message=sys_msg)
    memory2.add_message("What's the weather like?", role="user")
    memory2.add_message({"role": "assistant", "content": "It's sunny today!"})

    history2_full = memory2.get_history()
    logger.info(f"Memory2 Full History (with system message): {history2_full}")
    assert len(history2_full) == 3
    assert history2_full[0]["role"] == "system"
    assert history2_full[0]["content"] == sys_msg

    history2_limited = memory2.get_history(max_messages=1)
    logger.info(f"Memory2 Limited History (max_messages=1): {history2_limited}")
    assert len(history2_limited) == 2 # System message + 1 user/assistant message
    assert history2_limited[0]["role"] == "system"
    assert history2_limited[1]["content"] == "It's sunny today!"

    last_msg = memory2.get_last_message()
    logger.info(f"Memory2 Last Message: {last_msg}")
    assert last_msg and last_msg["content"] == "It's sunny today!"

    memory2.clear()
    history_after_clear = memory2.get_history()
    logger.info(f"Memory2 History after clear: {history_after_clear}")
    assert len(history_after_clear) == 1 # Should only contain system message
    assert history_after_clear[0]["role"] == "system"

    # Test changing system message
    memory2.set_system_message("You are a creative AI.")
    history_new_sys_msg = memory2.get_history()
    logger.info(f"Memory2 History after changing system message: {history_new_sys_msg}")
    assert len(history_new_sys_msg) == 1
    assert history_new_sys_msg[0]["content"] == "You are a creative AI."
    memory2.add_message("Be creative!", role="user")
    assert len(memory2.get_history()) == 2


    logger.info("SimpleMemory tests completed successfully.")

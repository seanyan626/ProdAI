# src/llms/base_llm.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model interactions.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs: Any):
        """
        Initializes the LLM.

        Args:
            model_name (str): The name of the model to use.
            api_key (Optional[str]): The API key for the LLM service.
            **kwargs: Additional keyword arguments for LLM configuration.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initializes the specific LLM client (e.g., OpenAI client).
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generates a text completion for a given prompt.

        Args:
            prompt (str): The input prompt.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            stop_sequences (Optional[List[str]]): Sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Asynchronously generates a text completion for a given prompt.

        Args:
            prompt (str): The input prompt.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            stop_sequences (Optional[List[str]]): Sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Returns:
            str: The generated text.
        """
        pass

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generates a chat completion for a given sequence of messages.
        This is a common pattern for chat models (e.g., GPT-3.5-turbo, GPT-4).

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries,
                e.g., [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}].
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            stop_sequences (Optional[List[str]]): Sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Dict[str, Any]: The full response from the LLM, typically including the message content and other metadata.
                            Example: {"role": "assistant", "content": "Response text"}

        Note: Subclasses should implement this if their underlying model supports chat completions.
              If not, they can raise NotImplementedError or adapt it.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support chat completions directly via this method. Implement it in the subclass.")

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Asynchronously generates a chat completion for a given sequence of messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            stop_sequences (Optional[List[str]]): Sequences to stop generation at.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Dict[str, Any]: The full response from the LLM.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support asynchronous chat completions directly via this method. Implement it in the subclass.")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the model.
        """
        return {
            "model_name": self.model_name,
            "config": self.config
        }

    # You might add other common methods here, like:
    # - count_tokens(text: str) -> int:
    # - get_embeddings(texts: List[str]) -> List[List[float]]: (though this might fit better in a separate Embeddings class)

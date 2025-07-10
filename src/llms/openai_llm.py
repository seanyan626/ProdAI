# src/llms/openai_llm.py
import logging
from typing import Any, Dict, List, Optional
import openai  # Using the official OpenAI library

from .base_llm import BaseLLM
from configs.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, load_config

# Ensure config is loaded when this module is imported or used.
load_config()

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    Wrapper for OpenAI's language models (GPT-3, GPT-3.5-turbo, GPT-4, etc.).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any
    ):
        """
        Initializes the OpenAI LLM.

        Args:
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-3.5-turbo", "text-davinci-003").
            api_key (Optional[str]): OpenAI API key. If None, tries to use OPENAI_API_KEY from config.
            max_tokens (int): Default maximum number of tokens for generation.
            temperature (float): Default sampling temperature.
            **kwargs: Additional arguments for the OpenAI client or model.
        """
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            logger.error("OpenAI API key is not set. Please provide it or set OPENAI_API_KEY in your .env file.")
            raise ValueError("OpenAI API key is missing.")

    def _initialize_client(self) -> None:
        """
        Initializes the OpenAI API client.
        The new OpenAI library (v1.0+) uses a client instance.
        """
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generates a text completion using a non-chat model (e.g., "text-davinci-003").
        For chat models like "gpt-3.5-turbo", use the `chat` method.
        """
        if self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4"):
            logger.warning(f"Model {self.model_name} is a chat model. Using `chat` method for generation.")
            # Adapt to chat format
            messages = [{"role": "user", "content": prompt}]
            chat_response = self.chat(messages, max_tokens, temperature, stop_sequences, **kwargs)
            return chat_response.get("content", "")

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        try:
            logger.debug(
                f"Generating text with OpenAI model {self.model_name}. Prompt: '{prompt[:100]}...'. "
                f"Max tokens: {current_max_tokens}, Temp: {current_temperature}"
            )
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,  # Pass other init-time configs
                **kwargs       # Pass runtime configs
            )
            generated_text = response.choices[0].text.strip()
            logger.debug(f"OpenAI response: '{generated_text[:100]}...'")
            return generated_text
        except openai.APIError as e:
            logger.error(f"OpenAI API error during generation: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI generation: {e}", exc_info=True)
            raise

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Asynchronously generates a text completion.
        """
        if self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4"):
            logger.warning(f"Async: Model {self.model_name} is a chat model. Using `achat` method.")
            messages = [{"role": "user", "content": prompt}]
            chat_response = await self.achat(messages, max_tokens, temperature, stop_sequences, **kwargs)
            return chat_response.get("content", "")

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        # The async client needs to be initialized if not already (or use openai.AsyncOpenAI)
        async_client = openai.AsyncOpenAI(api_key=self.api_key)

        try:
            logger.debug(
                f"Async generating text with OpenAI model {self.model_name}. Prompt: '{prompt[:100]}...'. "
                f"Max tokens: {current_max_tokens}, Temp: {current_temperature}"
            )
            response = await async_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            generated_text = response.choices[0].text.strip()
            logger.debug(f"Async OpenAI response: '{generated_text[:100]}...'")
            return generated_text
        except openai.APIError as e:
            logger.error(f"Async OpenAI API error during generation: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during async OpenAI generation: {e}", exc_info=True)
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generates a chat completion using a chat model (e.g., "gpt-3.5-turbo", "gpt-4").
        """
        if not (self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4") or "turbo" in self.model_name or "gpt-4o" in self.model_name):
            logger.warning(
                f"Model {self.model_name} may not be a chat model. "
                "The 'chat' method is intended for chat-based models. "
                "Consider using 'generate' for completion models."
            )

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        try:
            logger.debug(
                f"Generating chat completion with OpenAI model {self.model_name}. "
                f"Messages: {messages}. Max tokens: {current_max_tokens}, Temp: {current_temperature}"
            )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            # The response for chat completions is structured differently
            # response.choices[0].message is an ChatCompletionMessage object
            # It has attributes like .role and .content
            response_message = response.choices[0].message
            logger.debug(f"OpenAI chat response: Role: {response_message.role}, Content: '{str(response_message.content)[:100]}...'")
            return {"role": response_message.role, "content": response_message.content}
        except openai.APIError as e:
            logger.error(f"OpenAI API error during chat completion: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI chat completion: {e}", exc_info=True)
            raise

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Asynchronously generates a chat completion.
        """
        if not (self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4") or "turbo" in self.model_name or "gpt-4o" in self.model_name):
            logger.warning(
                f"Async: Model {self.model_name} may not be a chat model. "
                "The 'achat' method is intended for chat-based models."
            )

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        async_client = openai.AsyncOpenAI(api_key=self.api_key)

        try:
            logger.debug(
                f"Async generating chat completion with OpenAI model {self.model_name}. "
                f"Messages: {messages}. Max tokens: {current_max_tokens}, Temp: {current_temperature}"
            )
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            response_message = response.choices[0].message
            logger.debug(f"Async OpenAI chat response: Role: {response_message.role}, Content: '{str(response_message.content)[:100]}...'")
            return {"role": response_message.role, "content": response_message.content}
        except openai.APIError as e:
            logger.error(f"Async OpenAI API error during chat completion: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during async OpenAI chat completion: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # This part is for testing the OpenAILLM class
    # Ensure OPENAI_API_KEY is set in your .env file
    from configs.logging_config import setup_logging
    # Config is already loaded by the module's load_config() call at the top.
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY is not set or is a placeholder. Skipping OpenAILLM tests.")
    else:
        logger.info("Testing OpenAILLM...")

        # Test with a chat model
        try:
            chat_llm = OpenAILLM(model_name="gpt-3.5-turbo") # or DEFAULT_LLM_MODEL if it's a chat one
            logger.info(f"Chat LLM Info: {chat_llm.get_model_info()}")

            chat_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
            chat_response = chat_llm.chat(chat_messages)
            logger.info(f"Chat response for 'Capital of France': {chat_response['content']}")
            assert "paris" in chat_response['content'].lower()

            # Test generate with a chat model (should adapt)
            # generate_response_chat_model = chat_llm.generate("Translate 'hello' to French.")
            # logger.info(f"Generate response (from chat model) for 'Translate hello': {generate_response_chat_model}")
            # assert "bonjour" in generate_response_chat_model.lower()

        except Exception as e:
            logger.error(f"Error during OpenAILLM chat model test: {e}", exc_info=True)

        # Test with a completion model (e.g., gpt-3.5-turbo-instruct, if you have access or it's available)
        # Note: "text-davinci-003" is legacy. "gpt-3.5-turbo-instruct" is a newer instruct model.
        # If you don't have access to instruct models, this part might fail or need adjustment.
        # For now, we'll assume the user might not have easy access to non-chat completion models.
        # logger.info("Skipping non-chat completion model test as they are less common now.")
        # try:
        #     instruct_model_name = "gpt-3.5-turbo-instruct" # Or another completion model
        #     completion_llm = OpenAILLM(model_name=instruct_model_name)
        #     logger.info(f"Completion LLM Info: {completion_llm.get_model_info()}")
        #     completion_prompt = "Once upon a time,"
        #     completion_response = completion_llm.generate(completion_prompt, max_tokens=50)
        #     logger.info(f"Completion response for 'Once upon a time,': {completion_response}")
        #     assert len(completion_response) > 5
        # except openai.NotFoundError as e:
        #    logger.warning(f"Could not test instruct model {instruct_model_name}, it might not be available to your API key: {e}")
        # except Exception as e:
        #    logger.error(f"Error during OpenAILLM completion model test: {e}", exc_info=True)


        # Async tests (optional, can be run separately with an event loop)
        import asyncio
        async def run_async_tests():
            logger.info("Running async tests...")
            try:
                async_chat_llm = OpenAILLM(model_name="gpt-3.5-turbo")
                async_chat_messages = [
                    {"role": "user", "content": "What are three benefits of asynchronous programming?"}
                ]
                async_response = await async_chat_llm.achat(async_chat_messages, max_tokens=150)
                logger.info(f"Async chat response: {async_response['content']}")
                assert async_response['content']

                # Async generate with a chat model
                # async_gen_response = await async_chat_llm.agenerate("Tell me a short joke.")
                # logger.info(f"Async generate response (from chat model): {async_gen_response}")
                # assert async_gen_response

            except Exception as e:
                logger.error(f"Error during async OpenAILLM test: {e}", exc_info=True)

        logger.info("Attempting to run async tests...")
        try:
            asyncio.run(run_async_tests())
        except RuntimeError as e:
            # This can happen if an event loop is already running (e.g. in Jupyter)
            logger.warning(f"Could not run asyncio.run directly (possibly due to existing event loop): {e}. You might need to run async tests differently in this environment.")
        except Exception as e:
            logger.error(f"General error running async tests: {e}", exc_info=True)

        logger.info("OpenAILLM testing finished.")

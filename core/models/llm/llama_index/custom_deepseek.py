# core/models/llm/llama_index/custom_deepseek.py
import logging
from typing import Any, List, Optional

from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from openai import OpenAI
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class CustomDeepSeek(LLM):
    """
    一个与 LlamaIndex 兼容的自定义 DeepSeek LLM。
    """
    model_name: str
    api_key: str
    api_base: str
    temperature: float
    _client: OpenAI = PrivateAttr()

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model_name=model, api_key=api_key, api_base=api_base, temperature=temperature, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=api_base)

    @property
    def metadata(self) -> dict:
        return {"model_name": self.model_name}

    @llm_chat_callback()
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
            **kwargs,
        )
        return ChatResponse(
            message=ChatMessage(
                role="assistant", content=response.choices[0].message.content
            ),
            raw=response,
        )

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        raise NotImplementedError("stream_chat is not implemented for CustomDeepSeek")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        raise NotImplementedError("astream_chat is not implemented for CustomDeepSeek")

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("achat is not implemented for CustomDeepSeek")

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([ChatMessage(role="user", content=prompt)], **kwargs).message.content

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        return (await self.achat([ChatMessage(role="user", content=prompt)], **kwargs)).message.content

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("stream_complete is not implemented for CustomDeepSeek")

    async def astream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("astream_complete is not implemented for CustomDeepSeek")

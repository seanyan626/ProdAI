# core/models/llm/llama_index/custom_dashscope.py
import logging
from typing import Any, List, Optional

from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from dashscope import Generation

logger = logging.getLogger(__name__)


class CustomDashScope(LLM):
    """
    一个与 LlamaIndex 兼容的自定义 DashScope LLM。
    """
    model_name: str
    api_key: str
    temperature: float

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model_name=model_name, api_key=api_key, temperature=temperature, **kwargs)

    @property
    def metadata(self) -> dict:
        return {"model_name": self.model_name}

    @llm_chat_callback()
    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            api_key=self.api_key,
            temperature=self.temperature,
            **kwargs,
        )
        if response.status_code == 200:
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", content=response.output["text"]
                ),
                raw=response,
            )
        else:
            logger.error(f"DashScope LLM failed: {response.message}")
            raise RuntimeError(f"DashScope API error: {response.message}")

    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        # 一个简单的将 ChatMessage 列表转换为字符串的实现
        return "\n".join([f"{m.role}: {m.content}" for m in messages])

    # LlamaIndex 的 LLM 接口还需要 stream_chat 和 astream_chat 方法
    # 为了简单起见，我们暂时不实现它们

    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        raise NotImplementedError("stream_chat is not implemented for CustomDashScope")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs: Any):
        raise NotImplementedError("astream_chat is not implemented for CustomDashScope")

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("achat is not implemented for CustomDashScope")

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([ChatMessage(role="user", content=prompt)], **kwargs).message.content

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        return (await self.achat([ChatMessage(role="user", content=prompt)], **kwargs)).message.content

    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("stream_complete is not implemented for CustomDashScope")

    async def astream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("astream_complete is not implemented for CustomDashScope")

# core/models/llm/llama_index_llm.py
import logging
from abc import ABC, abstractmethod
from typing import Any, List

from llama_index.core.llms import LLM, ChatMessage

logger = logging.getLogger(__name__)


class BaseLlamaIndexLLM(ABC):
    """
    基于 LlamaIndex 的 LLM 模型的抽象基类。
    所有使用 LlamaIndex 框架的 LLM 都应继承此类。
    """

    def __init__(self, model_name: str, **kwargs):
        """
        初始化 LLM。

        参数:
            model_name (str): 要使用的 LLM 的名称。
            **kwargs: 其他特定于模型的配置参数。
        """
        self.model_name = model_name
        self.client: LLM = self._create_client(**kwargs)
        logger.info(f"LlamaIndex LLM 基类 '{self.__class__.__name__}' 使用模型 '{model_name}' 初始化。")

    @abstractmethod
    def _create_client(self, **kwargs) -> LLM:
        """
        创建并返回一个 LlamaIndex LLM 客户端实例。
        """
        pass

    def chat(self, messages: List[ChatMessage], **kwargs) -> Any:
        """
        使用提供的消息列表进行聊天。
        """
        return self.client.chat(messages, **kwargs)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> Any:
        """
        使用提供的消息列表进行异步聊天。
        """
        return await self.client.achat(messages, **kwargs)

    def get_model_info(self) -> dict:
        """
        返回关于 LLM 的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LlamaIndex",
            "type": "LLM"
        }

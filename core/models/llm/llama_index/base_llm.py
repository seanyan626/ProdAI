# core/models/llm/llama_index/base_llm.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llama_index.core.llms import LLM, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    基于 LlamaIndex 的 LLM 模型的抽象基类。
    所有使用 LlamaIndex 框架的 LLM 都应继承此类，以确保接口统一。
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        初始化 LLM。

        参数:
            model_name (str): 要使用的模型名称。
            **kwargs: 用于 LLM 配置的其他关键字参数。
        """
        self.model_name = model_name
        self.config = kwargs
        self.client: Optional[LLM] = None  # 客户端将在 _initialize_client 中被设置

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        初始化特定的 LlamaIndex LLM 客户端。
        此方法应由子类实现。
        """
        pass

    def get_client(self) -> LLM:
        """
        获取已初始化的客户端。如果未初始化，则抛出错误。
        """
        if not self.client:
            raise RuntimeError("LLM 客户端尚未初始化。请确保调用了 _initialize_client。")
        return self.client

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        使用提供的消息列表进行聊天。
        """
        client = self.get_client()
        return client.chat(messages, **kwargs)

    async def achat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        使用提供的消息列表进行异步聊天。
        """
        client = self.get_client()
        return await client.achat(messages, **kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        为给定的提示生成文本补全。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持 generate 方法。请使用 chat。")

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """
        异步为给定的提示生成文本补全。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持 agenerate 方法。请使用 achat。")

    def get_model_info(self) -> Dict[str, Any]:
        """
        返回关于模型的信息。
        """
        return {
            "model_name": self.model_name,
            "config": self.config,
            "framework": "LlamaIndex"
        }

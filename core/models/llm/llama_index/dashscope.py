# core/models/llm/llama_index/dashscope.py
from typing import Optional

try:
    from llama_index.llms.dashscope import DashScope as LlamaIndexDashScope
except ImportError:
    # 如果官方集成不存在，我们需要自己实现一个兼容的 LLM
    from .custom_dashscope import CustomDashScope as LlamaIndexDashScope

from llama_index.core.llms import LLM

from ..llama_index_llm import BaseLlamaIndexLLM
from configs.config import DASHSCOPE_API_KEY, DEFAULT_TEMPERATURE


class DashScopeLLM(BaseLlamaIndexLLM):
    """
    用于 DashScope 模型的 LlamaIndex LLM。
    """

    def __init__(
        self,
        model_name: str = "qwen-plus",
        api_key: Optional[str] = DASHSCOPE_API_KEY,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            **kwargs,
        )

    def _create_client(self, **kwargs) -> LLM:
        """
        创建并返回一个 LlamaIndex DashScope LLM 客户端实例。
        """
        return LlamaIndexDashScope(
            model_name=self.model_name,
            api_key=kwargs.pop("api_key"),
            temperature=kwargs.pop("temperature"),
            **kwargs,
        )

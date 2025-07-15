# core/models/llm/llama_index/deepseek.py
from typing import Optional

from .custom_deepseek import CustomDeepSeek as LlamaIndexDeepSeek

from llama_index.core.llms import LLM

from ..llama_index_llm import BaseLlamaIndexLLM
from configs.config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEFAULT_TEMPERATURE


class DeepSeekLLM(BaseLlamaIndexLLM):
    """
    用于 DeepSeek 模型的 LlamaIndex LLM。
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = DEEPSEEK_API_KEY,
        api_base: Optional[str] = DEEPSEEK_API_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            **kwargs,
        )

    def _create_client(self, **kwargs) -> LLM:
        """
        创建并返回一个 LlamaIndex DeepSeek LLM 客户端实例。
        """
        return LlamaIndexDeepSeek(
            model=self.model_name,
            api_key=kwargs.pop("api_key"),
            api_base=kwargs.pop("api_base"),
            temperature=kwargs.pop("temperature"),
            **kwargs,
        )

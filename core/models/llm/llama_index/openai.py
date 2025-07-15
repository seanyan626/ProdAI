# core/models/llm/llama_index/openai.py
from typing import Optional

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core.llms import LLM

from ..llama_index_llm import BaseLlamaIndexLLM
from configs.config import OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_TEMPERATURE


class OpenAILLM(BaseLlamaIndexLLM):
    """
    用于 OpenAI 模型的 LlamaIndex LLM。
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = OPENAI_API_KEY,
        api_base: Optional[str] = OPENAI_API_BASE,
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
        创建并返回一个 LlamaIndex OpenAI LLM 客户端实例。
        """
        return LlamaIndexOpenAI(
            model=self.model_name,
            api_key=kwargs.pop("api_key"),
            api_base=kwargs.pop("api_base"),
            temperature=kwargs.pop("temperature"),
            **kwargs,
        )

# core/models/llm/llama_index/openai_llm.py
import logging
from typing import Optional, Any

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from configs.config import OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, load_config
from .base_llm import BaseLLM

load_config()

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    用于 OpenAI 模型的 LlamaIndex LLM 封装。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        api_base: Optional[str] = OPENAI_API_BASE,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        **kwargs: Any
    ):
        all_kwargs = {
            "api_key": api_key,
            "api_base": api_base,
            "temperature": temperature,
            **kwargs
        }
        super().__init__(model_name=model_name, **all_kwargs)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        初始化 LlamaIndex OpenAI 客户端。
        """
        api_key = self.config.get("api_key")
        if not api_key:
            logger.error("OpenAI API 密钥未设置。")
            raise ValueError("OpenAI API 密钥缺失。")

        try:
            self.client = LlamaIndexOpenAI(
                model=self.model_name,
                api_key=api_key,
                api_base=self.config.get("api_base"),
                temperature=self.config.get("temperature", DEFAULT_TEMPERATURE),
                **self.config.get("llm_specific_kwargs", {})
            )
            logger.info(f"LlamaIndex OpenAI 客户端已为模型 {self.model_name} 初始化。")
        except Exception as e:
            logger.error(f"初始化 LlamaIndex OpenAI 客户端失败: {e}", exc_info=True)
            raise

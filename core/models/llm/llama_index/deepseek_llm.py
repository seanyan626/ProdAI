# core/models/llm/llama_index/deepseek_llm.py
import logging
from typing import Optional, Any

# 使用 OpenAILike 来封装兼容 OpenAI API 的模型
from llama_index.llms.openai_like import OpenAILike

from configs.config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL_NAME, load_config
from .base_llm import BaseLLM

load_config()

logger = logging.getLogger(__name__)


class DeepSeekLLM(BaseLLM):
    """
    用于 DeepSeek 模型的 LlamaIndex LLM 封装。
    使用 OpenAILike 实现，因为它兼容 OpenAI API。
    """

    def __init__(
        self,
        model_name: str = DEEPSEEK_MODEL_NAME,
        api_key: Optional[str] = DEEPSEEK_API_KEY,
        api_base: Optional[str] = DEEPSEEK_API_URL,
        temperature: Optional[float] = 0.7,
        **kwargs: Any
    ):
        all_kwargs = {
            "api_key": api_key,
            "api_base": api_base,
            "temperature": temperature,
            "is_chat_model": True,  # 明确指定为聊天模型
            **kwargs
        }
        super().__init__(model_name=model_name, **all_kwargs)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        初始化 LlamaIndex OpenAILike 客户端以调用 DeepSeek。
        """
        api_key = self.config.get("api_key")
        if not api_key:
            logger.error("DeepSeek API 密钥 (DEEPSEEK_API_KEY) 未设置。")
            raise ValueError("DeepSeek API 密钥缺失。")

        try:
            self.client = OpenAILike(
                model=self.model_name,
                api_key=api_key,
                api_base=self.config.get("api_base"),
                temperature=self.config.get("temperature", 0.7),
                is_chat_model=self.config.get("is_chat_model", True),
                **self.config.get("llm_specific_kwargs", {})
            )
            logger.info(f"LlamaIndex OpenAILike (for DeepSeek) 客户端已为模型 {self.model_name} 初始化。")
        except Exception as e:
            logger.error(f"初始化 LlamaIndex OpenAILike (for DeepSeek) 客户端失败: {e}", exc_info=True)
            raise

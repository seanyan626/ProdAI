# core/models/embedding/langchain/openai_embedding.py
import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings

from configs.config import OPENAI_API_KEY, OPENAI_API_BASE

logger = logging.getLogger(__name__)


class OpenAIEmbedding(LangChainOpenAIEmbeddings):
    """
    用于 OpenAI 模型的 LangChain Embedding 封装。
    直接继承自 LangChain 的官方实现，以获得最佳兼容性。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = OPENAI_API_KEY,
        api_base: Optional[str] = OPENAI_API_BASE,
        **kwargs,
    ):
        """
        初始化 OpenAI 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): OpenAI API 密钥。
            api_base (str, optional): OpenAI API 端点。
        """
        # 将参数转换为 LangChain 客户端接受的格式
        super().__init__(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            **kwargs,
        )
        logger.info(f"LangChain OpenAIEmbedding initialized with model '{model_name}'.")

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model,
            "framework": "LangChain",
            "type": "Embedding"
        }

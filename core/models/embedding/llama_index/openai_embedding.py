# core/models/embedding/llama_index/openai_embedding.py
import logging
from typing import Optional

from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbedding

from configs.config import config

logger = logging.getLogger(__name__)


class OpenAIEmbedding(LlamaIndexOpenAIEmbedding):
    """
    用于 OpenAI 模型的 LlamaIndex Embedding 封装。
    直接继承自 LlamaIndex 的官方实现，以获得最佳兼容性。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 OpenAI 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): OpenAI API 密钥。
            api_base (str, optional): OpenAI API 端点。
        """
        # 使用配置中的值作为默认值
        model_name = model_name or config.openai.embedding_name or "text-embedding-ada-002"
        api_key = api_key or config.openai.embedding_api_key or config.openai.api_key
        api_base = api_base or config.openai.embedding_api_url or config.openai.api_base
        
        super().__init__(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        logger.info(f"LlamaIndex OpenAIEmbedding initialized with model '{model_name}'.")

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model,
            "framework": "LlamaIndex",
            "type": "Embedding",
            "provider": "OpenAI"
        }

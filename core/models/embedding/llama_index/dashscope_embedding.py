# core/models/embedding/llama_index/dashscope_embedding.py
import logging
from typing import Optional

# DashScope 的 Embedding API 与 OpenAI 兼容，因此我们可以复用 OpenAIEmbedding 类
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbedding

from configs.config import DASHSCOPE_API_KEY, DASHSCOPE_API_URL

logger = logging.getLogger(__name__)

# DashScope 提供的文本向量模型
DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v2"


class DashScopeEmbedding(LlamaIndexOpenAIEmbedding):
    """
    用于 DashScope (通义千问) 模型的 LlamaIndex Embedding 封装。
    由于其 API 与 OpenAI 兼容，我们直接继承并配置 LlamaIndex 的 OpenAIEmbedding 类。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
        api_key: Optional[str] = DASHSCOPE_API_KEY,
        api_base: Optional[str] = DASHSCOPE_API_URL,
        **kwargs,
    ):
        """
        初始化 DashScope 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): DashScope API 密钥。
            api_base (str, optional): DashScope API 端点。
        """
        if not api_key:
            raise ValueError("DashScope API key is required.")

        # 使用 OpenAIEmbedding 的初始化方法，但传入 DashScope 的参数
        super().__init__(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )
        logger.info(f"LlamaIndex DashScopeEmbedding (via OpenAIEmbedding) initialized with model '{model_name}'.")

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model,
            "framework": "LlamaIndex",
            "type": "Embedding"
        }

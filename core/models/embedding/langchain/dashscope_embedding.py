# core/models/embedding/langchain/dashscope_embedding.py
import logging
from typing import List, Optional

from dashscope import TextEmbedding
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse

from .base_embedding import BaseEmbedding
from configs.config import DASHSCOPE_API_KEY

logger = logging.getLogger(__name__)


class DashScopeEmbedding(BaseEmbedding):
    """
    用于 DashScope (通义千问) 模型的 LangChain Embedding 封装。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v1",
        api_key: Optional[str] = DASHSCOPE_API_KEY,
        **kwargs,
    ):
        """
        初始化 DashScope 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): DashScope API 密钥。
        """
        super().__init__(model_name=model_name, **kwargs)
        self.api_key = api_key

        if not self.api_key:
            raise ValueError("DashScope API key is required.")

        logger.info(f"LangChain DashScopeEmbedding initialized with model '{model_name}'.")

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用 DashScope API 并处理响应。
        """
        response: DashScopeAPIResponse = TextEmbedding.call(
            model=self.model_name,
            input=texts,
            api_key=self.api_key,
        )
        if response.status_code == 200:
            return [embedding["embedding"] for embedding in response.output["embeddings"]]
        else:
            logger.error(f"DashScope embedding failed: {response.message}")
            raise RuntimeError(f"DashScope API error: {response.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档生成嵌入。
        """
        return self._call_embedding_api(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入。
        """
        return self._call_embedding_api([text])[0]

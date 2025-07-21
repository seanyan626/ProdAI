# core/models/embedding/llama_index/dashscope_embedding.py
import logging
from typing import Optional, List

from dashscope import TextEmbedding
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from llama_index.core.embeddings import BaseEmbedding

from configs.config import config

logger = logging.getLogger(__name__)

# DashScope 提供的文本向量模型
DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v1"


class DashScopeEmbedding(BaseEmbedding):
    """
    用于 DashScope (通义千问) 模型的 LlamaIndex Embedding 封装。
    使用 DashScope 原生 SDK，正确继承 BaseEmbedding。
    """

    # 定义 Pydantic 模型字段
    model_name: str = DEFAULT_DASHSCOPE_EMBEDDING_MODEL
    api_key: Optional[str] = None

    def __init__(
            self,
            model_name: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs,
    ):
        """
        初始化 DashScope 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): DashScope API 密钥。
        """

        # 使用配置中的值作为默认值
        actual_model_name = model_name or config.dashscope.embedding_model_name or DEFAULT_DASHSCOPE_EMBEDDING_MODEL
        actual_api_key = api_key or config.dashscope.api_key

        if not actual_api_key:
            raise ValueError("DashScope API key is required.")

        # 正确初始化 Pydantic 模型
        super().__init__(
            model_name=actual_model_name,
            api_key=actual_api_key,
            **kwargs
        )

        logger.info(f"LlamaIndex DashScopeEmbedding initialized with model '{self.model_name}'.")

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        获取查询文本的嵌入向量。
        """
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量。
        """
        try:
            response: DashScopeAPIResponse = TextEmbedding.call(
                model=self.model_name,
                input=[text],
                api_key=self.api_key,
            )

            if response.status_code == 200:
                embeddings = response.output["embeddings"]
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]["embedding"]
                else:
                    raise RuntimeError("No embeddings returned from DashScope API")
            else:
                logger.error(f"DashScope embedding failed: {response.message}")
                raise RuntimeError(f"DashScope API error: {response.message}")

        except Exception as e:
            logger.error(f"Error calling DashScope embedding API: {str(e)}")
            raise

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        异步获取查询文本的嵌入向量。
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, query)

    def get_text_embedding(self, text: str) -> List[float]:
        """
        公共方法：获取文本的嵌入向量。
        """
        return self._get_text_embedding(text)

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LlamaIndex",
            "type": "Embedding",
            "provider": "DashScope"
        }

# core/models/embedding/openai_embedding_model.py
# 使用 Langchain 封装 OpenAI 文本嵌入模型的实现
import logging
from typing import List, Optional, Any

from langchain_openai import OpenAIEmbeddings

from .base_embedding_model import BaseEmbeddingModel
from configs.config import OPENAI_API_KEY, OPENAI_API_BASE, load_config

load_config()

logger = logging.getLogger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    使用 Langchain 的 OpenAIEmbeddings 实现的文本嵌入模型。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = OPENAI_API_KEY,
        base_url: Optional[str] = OPENAI_API_BASE,
        **kwargs: Any
    ):
        """
        初始化 OpenAI 嵌入模型。
        """
        all_kwargs = {"base_url": base_url, **kwargs}
        super().__init__(model_name=model_name, **all_kwargs)
        self.api_key = api_key
        self.client: Optional[OpenAIEmbeddings] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        初始化 Langchain OpenAIEmbeddings 客户端。
        """
        if not self.api_key:
            logger.error("OpenAI API 密钥未设置。请提供密钥或在 .env 文件中设置 OPENAI_API_KEY。")
            raise ValueError("OpenAI API 密钥缺失。")
        try:
            client_params = {
                "model": self.model_name,
                "openai_api_key": self.api_key,
                "openai_api_base": self.config.get("base_url"),
                **(self.config.get("embedding_specific_kwargs") or {})
            }
            client_params = {k: v for k, v in client_params.items() if v is not None}

            self.client = OpenAIEmbeddings(**client_params)
            logger.info(f"Langchain OpenAIEmbeddings 客户端已为模型 '{self.model_name}' 初始化。")
        except Exception as e:
            logger.error(f"初始化 Langchain OpenAIEmbeddings 客户端失败: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档（文本列表）生成嵌入向量。
        """
        if not self.client:
            raise RuntimeError("嵌入客户端未初始化。")
        if not texts:
            return []
        try:
            logger.debug(f"正在为 {len(texts)} 个文档生成嵌入...")
            embeddings = self.client.embed_documents(texts)
            logger.info(f"成功为 {len(texts)} 个文档生成嵌入。")
            return embeddings
        except Exception as e:
            logger.error(f"使用 OpenAIEmbeddings 为文档生成嵌入失败: {e}", exc_info=True)
            return [[] for _ in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步为一批文档（文本列表）生成嵌入向量。
        """
        if not self.client:
            raise RuntimeError("嵌入客户端未初始化。")
        if not texts:
            return []
        try:
            logger.debug(f"正在异步为 {len(texts)} 个文档生成嵌入...")
            embeddings = await self.client.aembed_documents(texts)
            logger.info(f"成功异步为 {len(texts)} 个文档生成嵌入。")
            return embeddings
        except Exception as e:
            logger.error(f"使用 OpenAIEmbeddings 异步为文档生成嵌入失败: {e}", exc_info=True)
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量。
        """
        if not self.client:
            raise RuntimeError("嵌入客户端未初始化。")
        if not text:
            logger.warning("embed_query 接收到空文本输入，返回空嵌入。")
            return []
        try:
            logger.debug(f"正在为查询 '{text[:50]}...' 生成嵌入...")
            embedding = self.client.embed_query(text)
            logger.info(f"成功为查询生成嵌入。")
            return embedding
        except Exception as e:
            logger.error(f"使用 OpenAIEmbeddings 为查询生成嵌入失败: {e}", exc_info=True)
            return []

    async def aembed_query(self, text: str) -> List[float]:
        """
        异步为单个查询文本生成嵌入向量。
        """
        if not self.client:
            raise RuntimeError("嵌入客户端未初始化。")
        if not text:
            logger.warning("aembed_query 接收到空文本输入，返回空嵌入。")
            return []
        try:
            logger.debug(f"正在异步为查询 '{text[:50]}...' 生成嵌入...")
            embedding = await self.client.aembed_query(text)
            logger.info(f"成功异步为查询生成嵌入。")
            return embedding
        except Exception as e:
            logger.error(f"使用 OpenAIEmbeddings 异步为查询生成嵌入失败: {e}", exc_info=True)
            return []

    def get_langchain_client(self) -> Any:
        """
        返回底层的 Langchain OpenAIEmbeddings 实例。
        """
        return self.client

if __name__ == '__main__':
    import asyncio
    from configs.logging_config import setup_logging

    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OpenAI API 密钥未在 .env 文件中配置或为占位符，跳过 OpenAIEmbeddingModel 的 __main__ 测试。")
    else:
        logger.info("测试 OpenAIEmbeddingModel...")
        try:
            embedding_model = OpenAIEmbeddingModel()
            logger.info(f"嵌入模型信息: {embedding_model.get_model_info()}")

            query_text = "这是一个示例文本查询。"
            query_embedding = embedding_model.embed_query(query_text)
            logger.info(f"查询 '{query_text}' 的嵌入向量 (前5个维度): {query_embedding[:5]}")
            assert isinstance(query_embedding, list) and len(query_embedding) > 0

            doc_texts = ["第一份文档。", "第二份文档。"]
            doc_embeddings = embedding_model.embed_documents(doc_texts)
            logger.info(f"为 {len(doc_texts)} 个文档生成的嵌入向量 (第一个文档的前5个维度): {doc_embeddings[0][:5]}")
            assert isinstance(doc_embeddings, list) and len(doc_embeddings) == len(doc_texts)

            async def run_async_embedding_tests():
                logger.info("\n--- 测试异步嵌入方法 ---")
                async_query_embedding = await embedding_model.aembed_query(query_text + " (异步)")
                logger.info(f"异步查询嵌入 (前5个维度): {async_query_embedding[:5]}")
                assert len(async_query_embedding) > 0

                async_doc_embeddings = await embedding_model.aembed_documents([d + " (异步)" for d in doc_texts])
                logger.info(f"异步文档嵌入 (第一个文档的前5个维度): {async_doc_embeddings[0][:5]}")
                assert len(async_doc_embeddings) == len(doc_texts)

            asyncio.run(run_async_embedding_tests())
            logger.info("异步嵌入测试完成。")

        except Exception as e:
            logger.error(f"执行 OpenAIEmbeddingModel 测试时发生意外错误: {e}", exc_info=True)

        logger.info("OpenAIEmbeddingModel __main__ 测试结束。")
    pass

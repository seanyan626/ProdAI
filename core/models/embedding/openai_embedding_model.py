# core/models/embedding/openai_embedding_model.py
# 使用 Langchain 封装 OpenAI 文本嵌入模型的实现
import logging
from typing import List, Optional, Any

from langchain_openai import OpenAIEmbeddings

from .base_embedding_model import BaseEmbeddingModel

from configs.config import OPENAI_API_KEY, OPENAI_API_BASE, load_config

load_config()

from configs.config import OPENAI_API_KEY # 假设 API 密钥从这里获取


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

        model_name: str = "text-embedding-ada-002", # OpenAI 推荐的默认模型
        api_key: Optional[str] = OPENAI_API_KEY,
        **kwargs: Any # 其他传递给 OpenAIEmbeddings 的参数
    ):
        """
        初始化 OpenAI 嵌入模型。

        参数:
            model_name (str): 要使用的 OpenAI 嵌入模型的名称。
            api_key (Optional[str]): OpenAI API 密钥。如果为 None，则尝试从配置中获取。
            **kwargs: 其他传递给 langchain_openai.OpenAIEmbeddings 的参数。
        """
        super().__init__(model_name=model_name, **kwargs) # 将kwargs传递给基类存储在config中

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

                "model": self.model_name, # OpenAIEmbeddings 使用 'model' 而不是 'model_name'
                "openai_api_key": self.api_key,
                **(self.config.get("embedding_specific_kwargs") or {}) # 从config传入的额外参数
            }
            # 移除kwargs中可能与基类构造函数冲突或不被OpenAIEmbeddings直接接受的参数
            # 例如，如果基类也接受了 'temperature' 等不适用于 embedding 的参数
            # 不过这里 BaseEmbeddingModel 的 __init__ 比较简单，主要冲突是 model_name vs model

            # OpenAIEmbeddings 的构造函数参数比较直接，不像 ChatOpenAI 那么多变
            # 主要参数是 model, openai_api_key, openai_api_base, chunk_size 等
            # kwargs 应该直接透传给它，由Pydantic处理。
            # 我们确保 model 和 openai_api_key 被正确设置。

            # 从 self.config 中提取并覆盖 client_params (如果用户在kwargs中指定了model或openai_api_key)
            # 但通常这些应该通过命名参数传入。
            # 这里假设 self.config 主要用于存储基类未直接处理的、特定于此实现的额外参数。

            # 构造函数参数的优先级: 显式参数 > kwargs中的参数 (通过self.config) > 默认值
            # OpenAIEmbeddings 的参数包括： model, deployment (for Azure), openai_api_version, etc.
            # 我们将基类 kwargs 中不属于 BaseEmbeddingModel 定义的参数传递给 OpenAIEmbeddings

            extra_lc_kwargs = self.config.copy() # 复制一份config
            # 移除已在BaseEmbeddingModel中处理或OpenAIEmbeddingModel构造函数中显式使用的参数
            extra_lc_kwargs.pop("model_name", None) # model_name 已用于 client_params["model"]

            final_client_params = {**client_params, **extra_lc_kwargs}

            self.client = OpenAIEmbeddings(**final_client_params)

            logger.info(f"Langchain OpenAIEmbeddings 客户端已为模型 '{self.model_name}' 初始化。")
        except Exception as e:
            logger.error(f"初始化 Langchain OpenAIEmbeddings 客户端失败: {e}", exc_info=True)
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档（文本列表）生成嵌入向量。
        """
        if not self.client:


            logger.error("OpenAIEmbeddings 客户端未初始化。")

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

            # 根据需要，可以返回空列表或重新引发自定义异常
            # raise EmbeddingGenerationError(f"Failed to embed documents: {e}") from e
            return [[] for _ in texts] # 返回与输入文本数量相同的空列表，表示失败


    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步为一批文档（文本列表）生成嵌入向量。
        """
        if not self.client:

            logger.error("OpenAIEmbeddings 客户端未初始化。")

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

            logger.error("OpenAIEmbeddings 客户端未初始化。")
            raise RuntimeError("嵌入客户端未初始化。")
        if not text: # 处理空字符串输入

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

            logger.error("OpenAIEmbeddings 客户端未初始化。")

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

if __name__ == '__main__':
    import asyncio

    from configs.logging_config import setup_logging


    from configs.config import load_config # OPENAI_API_KEY 从这里导入
    from configs.logging_config import setup_logging

    load_config()

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


            # 默认模型 text-embedding-ada-002
            embedding_model = OpenAIEmbeddingModel()
            logger.info(f"嵌入模型信息: {embedding_model.get_model_info()}")

            # 测试 embed_query
            query_text = "这是一个示例文本查询。"
            query_embedding = embedding_model.embed_query(query_text)
            logger.info(f"查询 '{query_text}' 的嵌入向量 (前5个维度): {query_embedding[:5]}")
            assert isinstance(query_embedding, list)
            assert len(query_embedding) > 0 # OpenAI embedding 通常是 1536 维
            assert isinstance(query_embedding[0], float)

            # 测试 embed_documents
            doc_texts = [
                "第一份文档是关于人工智能的。",
                "第二份文档讨论了自然语言处理。",
                "这是第三份文档，内容简短。"
            ]
            doc_embeddings = embedding_model.embed_documents(doc_texts)
            logger.info(f"为 {len(doc_texts)} 个文档生成的嵌入向量 (第一个文档的前5个维度): {doc_embeddings[0][:5]}")
            assert isinstance(doc_embeddings, list)
            assert len(doc_embeddings) == len(doc_texts)
            assert isinstance(doc_embeddings[0], list)
            assert len(doc_embeddings[0]) == len(query_embedding) # 所有嵌入应具有相同维度
            assert isinstance(doc_embeddings[0][0], float)

            # 测试异步方法

            async def run_async_embedding_tests():
                logger.info("\n--- 测试异步嵌入方法 ---")
                async_query_embedding = await embedding_model.aembed_query(query_text + " (异步)")
                logger.info(f"异步查询嵌入 (前5个维度): {async_query_embedding[:5]}")

                assert len(async_query_embedding) > 0

                assert len(async_query_embedding) == len(query_embedding)


                async_doc_embeddings = await embedding_model.aembed_documents([d + " (异步)" for d in doc_texts])
                logger.info(f"异步文档嵌入 (第一个文档的前5个维度): {async_doc_embeddings[0][:5]}")
                assert len(async_doc_embeddings) == len(doc_texts)

                assert len(async_doc_embeddings[0]) == len(query_embedding)


            asyncio.run(run_async_embedding_tests())
            logger.info("异步嵌入测试完成。")


        except ValueError as ve: # API Key 缺失等初始化错误
            logger.error(f"OpenAIEmbeddingModel 初始化或配置错误: {ve}")

        except Exception as e:
            logger.error(f"执行 OpenAIEmbeddingModel 测试时发生意外错误: {e}", exc_info=True)

        logger.info("OpenAIEmbeddingModel __main__ 测试结束。")
    pass

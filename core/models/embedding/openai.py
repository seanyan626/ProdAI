# core/models/embedding/openai.py
import logging
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbedding

from .langchain_embedding import BaseLangChainEmbeddingModel
from .llama_index_embedding import BaseLlamaIndexEmbeddingModel
from configs.config import OPENAI_API_KEY, OPENAI_API_BASE

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseLangChainEmbeddingModel, BaseLlamaIndexEmbeddingModel):
    """
    OpenAI 嵌入模型，同时支持 LangChain 和 LlamaIndex 框架。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = OPENAI_API_KEY,
        api_base: Optional[str] = OPENAI_API_BASE,
        framework: str = "langchain",  # "langchain" or "llama_index"
        **kwargs,
    ):
        """
        初始化 OpenAI 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): OpenAI API 密钥。
            api_base (str, optional): OpenAI API 端点。
            framework (str): 要使用的框架 ("langchain" 或 "llama_index")。
        """
        super().__init__(model_name=model_name, **kwargs)
        self.framework = framework

        if self.framework == "langchain":
            self.client = LangChainOpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                **kwargs,
            )
        elif self.framework == "llama_index":
            self.client = LlamaIndexOpenAIEmbedding(
                model=model_name,
                api_key=api_key,
                api_base=api_base,
                **kwargs,
            )
        else:
            raise ValueError("Unsupported framework. Choose 'langchain' or 'llama_index'.")

        logger.info(f"OpenAIEmbedding for {framework} initialized with model '{model_name}'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.framework != "langchain":
            raise NotImplementedError("This method is only for the LangChain framework.")
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.framework != "langchain":
            raise NotImplementedError("This method is only for the LangChain framework.")
        return self.client.embed_query(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        if self.framework != "llama_index":
            raise NotImplementedError("This method is only for the LlamaIndex framework.")
        return self.client._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.framework != "llama_index":
            raise NotImplementedError("This method is only for the LlamaIndex framework.")
        return self.client._get_text_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        if self.framework != "llama_index":
            raise NotImplementedError("This method is only for the LlamaIndex framework.")
        return await self.client._aget_query_embedding(query)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["framework"] = self.framework
        return info

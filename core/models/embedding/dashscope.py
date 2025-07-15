# core/models/embedding/dashscope.py
import logging
from typing import List, Optional

from dashscope import TextEmbedding

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse


from .langchain_embedding import BaseLangChainEmbeddingModel
from .llama_index_embedding import BaseLlamaIndexEmbeddingModel
from configs.config import DASHSCOPE_API_KEY

logger = logging.getLogger(__name__)


class DashScopeEmbedding(BaseLangChainEmbeddingModel, BaseLlamaIndexEmbeddingModel):
    """
    DashScope (通义千问) 嵌入模型，同时支持 LangChain 和 LlamaIndex 框架。
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v1",
        api_key: Optional[str] = DASHSCOPE_API_KEY,
        framework: str = "langchain",  # "langchain" or "llama_index"
        **kwargs,
    ):
        """
        初始化 DashScope 嵌入模型。

        参数:
            model_name (str): 模型名称。
            api_key (str, optional): DashScope API 密钥。
            framework (str): 要使用的框架 ("langchain" 或 "llama_index")。
        """
        super().__init__(model_name=model_name, **kwargs)
        self.framework = framework
        self.api_key = api_key

        if not self.api_key:
            raise ValueError("DashScope API key is required.")

        logger.info(f"DashScopeEmbedding for {framework} initialized with model '{model_name}'.")

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:

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
        if self.framework != "langchain":
            raise NotImplementedError("This method is only for the LangChain framework.")
        return self._call_embedding_api(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.framework != "langchain":
            raise NotImplementedError("This method is only for the LangChain framework.")
        return self._call_embedding_api([text])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        if self.framework != "llama_index":
            raise NotImplementedError("This method is only for the LlamaIndex framework.")
        return self._call_embedding_api([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.framework != "llama_index":
            raise NotImplementedError("This method is only for the LlamaIndex framework.")
        return self._call_embedding_api(texts)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["framework"] = self.framework
        return info

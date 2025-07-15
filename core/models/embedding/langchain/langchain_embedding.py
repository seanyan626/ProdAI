# core/models/embedding/langchain_embedding.py
import logging
from abc import abstractmethod
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BaseLangChainEmbeddingModel(Embeddings):
    """
    基于 LangChain 的文本嵌入模型的抽象基类。
    所有使用 LangChain 框架的嵌入模型都应继承此类。
    """

    def __init__(self, model_name: str, **kwargs):
        """
        初始化嵌入模型。

        参数:
            model_name (str): 要使用的嵌入模型的名称。
            **kwargs: 传递给 LangChain `Embeddings` 基类的其他参数。
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        logger.info(f"LangChain 嵌入模型基类 '{self.__class__.__name__}' 使用模型 '{model_name}' 初始化。")

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档生成嵌入。
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入。
        """
        pass

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LangChain",
            "type": "EmbeddingModel"
        }

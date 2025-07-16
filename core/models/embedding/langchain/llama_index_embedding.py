# core/models/embedding/llama_index_embedding.py
import logging
from abc import abstractmethod
from typing import List


from llama_index.core.base.embeddings.base import BaseEmbedding


logger = logging.getLogger(__name__)


class BaseLlamaIndexEmbeddingModel(BaseEmbedding):
    """
    基于 LlamaIndex 的文本嵌入模型的抽象基类。
    所有使用 LlamaIndex 框架的嵌入模型都应继承此类。
    """

    def __init__(self, model_name: str, **kwargs):
        """
        初始化嵌入模型。

        参数:
            model_name (str): 要使用的嵌入模型的名称。
            **kwargs: 传递给 LlamaIndex `BaseEmbedding` 基类的其他参数。
        """
        super().__init__(model_name=model_name, **kwargs)
        logger.info(f"LlamaIndex 嵌入模型基类 '{self.__class__.__name__}' 使用模型 '{model_name}' 初始化。")

    @abstractmethod
    def _get_text_embedding(self, text: str) -> List[float]:
        """
        为单个文本生成嵌入。
        """
        pass

    @abstractmethod
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本生成嵌入。
        """
        pass

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        为查询文本生成嵌入。
        LlamaIndex 的基类默认会调用 _get_text_embedding。
        """
        return self._get_text_embedding(query)

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LlamaIndex",
            "type": "EmbeddingModel"
        }

# core/models/embedding/langchain/base_embedding.py
from abc import ABC
from typing import Dict, Any
from langchain_core.embeddings import Embeddings


class BaseEmbedding(Embeddings, ABC):
    """
    基于 LangChain 的文本嵌入模型的抽象基类。
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        初始化嵌入模型。

        参数:
            model_name (str): 要使用的嵌入模型的名称。
            **kwargs: 其他特定于模型的配置参数。
        """
        super().__init__(**kwargs)
        self.model_name = model_name

    def get_model_info(self) -> Dict[str, Any]:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LangChain",
            "type": "Embedding"
        }

# core/models/embedding/llama_index/base_embedding.py
from abc import ABC
from typing import Dict, Any
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaIndexBaseEmbedding


class BaseEmbedding(LlamaIndexBaseEmbedding, ABC):
    """
    基于 LlamaIndex 的文本嵌入模型的抽象基类。
    """

    def __init__(self, model_name: str, **kwargs: Any):
        """
        初始化嵌入模型。

        参数:
            model_name (str): 要使用的嵌入模型的名称。
            **kwargs: 其他特定于模型的配置参数。
        """
        super().__init__(model_name=model_name, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "framework": "LlamaIndex",
            "type": "Embedding"
        }

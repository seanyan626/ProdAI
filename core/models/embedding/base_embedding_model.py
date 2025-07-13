# core/models/embedding/base_embedding_model.py
# 文本嵌入模型的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import List, Union

logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    """
    文本嵌入模型的抽象基类。
    子类应实现与特定嵌入服务或库交互的具体逻辑。
    """

    def __init__(self, model_name: str, **kwargs):
        """
        初始化嵌入模型。

        参数:
            model_name (str): 要使用的嵌入模型的名称。
            **kwargs: 其他特定于模型的配置参数。
        """
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"嵌入模型基类 '{self.__class__.__name__}' 使用模型 '{model_name}' 初始化。")

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档（文本列表）生成嵌入向量。

        参数:
            texts (List[str]): 需要转换为嵌入向量的文本列表。

        返回:
            List[List[float]]: 每个输入文本对应的嵌入向量列表。
                               列表中的每个向量本身也是一个浮点数列表。
        """
        pass

    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步为一批文档（文本列表）生成嵌入向量。

        参数:
            texts (List[str]): 需要转换为嵌入向量的文本列表。

        返回:
            List[List[float]]: 每个输入文本对应的嵌入向量列表。
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入向量。
        通常用于相似性搜索中的查询向量化。

        参数:
            text (str): 需要转换为嵌入向量的查询文本。

        返回:
            List[float]: 查询文本对应的嵌入向量。
        """
        pass

    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """
        异步为单个查询文本生成嵌入向量。

        参数:
            text (str): 需要转换为嵌入向量的查询文本。

        返回:
            List[float]: 查询文本对应的嵌入向量。
        """
        pass

    def get_model_info(self) -> dict:
        """
        返回关于嵌入模型的信息。
        """
        return {
            "model_name": self.model_name,
            "config": self.config,
            "type": "EmbeddingModel"
        }

if __name__ == '__main__':
    logger.info("BaseEmbeddingModel 模块。这是一个抽象基类，不应直接实例化。")
    # 示例：
    # class MyEmbeddingModel(BaseEmbeddingModel):
    #     def embed_documents(self, texts: List[str]) -> List[List[float]]:
    #         # 模拟实现
    #         logger.info(f"正在为 {len(texts)} 个文档生成嵌入...")
    #         return [[float(i) for i in range(texts[0].count('a'))]] * len(texts) # 示例逻辑
    #
    #     async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #         logger.info(f"正在异步为 {len(texts)} 个文档生成嵌入...")
    #         return await asyncio.to_thread(self.embed_documents, texts)
    #
    #     def embed_query(self, text: str) -> List[float]:
    #         logger.info(f"正在为查询 '{text[:30]}...' 生成嵌入...")
    #         return [float(text.count('a'))] # 示例逻辑
    #
    #     async def aembed_query(self, text: str) -> List[float]:
    #         logger.info(f"正在异步为查询 '{text[:30]}...' 生成嵌入...")
    #         return await asyncio.to_thread(self.embed_query, text)
    #
    # # my_model = MyEmbeddingModel(model_name="my-test-embedding-v1")
    # # print(my_model.get_model_info())
    # # print(my_model.embed_query("banana apple"))
    pass

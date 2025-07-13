# core/rag/base_retriever.py
# RAG 系统中文档检索器的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Document:
    """
    表示一个文档的简单数据结构，包含页面内容和元数据。
    """
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self.page_content = page_content # 页面内容
        self.metadata = metadata or {}  # 元数据
        # 允许将任意附加字段作为元数据的一部分，以提高灵活性
        if kwargs:
            self.metadata.update(kwargs)

    def __repr__(self):
        return f"Document(页面内容='{self.page_content[:50]}...', 元数据={self.metadata})"

class BaseRetriever(ABC):
    """
    RAG 系统中文档检索器的抽象基类。
    子类应实现具体的文档检索逻辑。
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        为给定查询检索相关文档。
        （实现待补充）
        """
        pass

    async def aretrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        异步为给定查询检索相关文档。
        （实现待补充，或默认调用同步版本）
        """
        # logger.warning(
        #     f"检索器 '{self.__class__.__name__}' 未实现特定的异步版本。"
        #     "将回退到同步检索。"
        # )
        # return self.retrieve(query, top_k, **kwargs)
        pass

    def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        将文档添加到检索器的底层向量存储或索引中。
        （实现待补充，并非所有检索器都支持）
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持直接添加文档。")

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        异步将文档添加到检索器的底层向量存储或索引中。
        （实现待补充，并非所有检索器都支持）
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持异步直接添加文档。")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
    logger.info("BaseRetriever 模块。这是一个抽象基类，通常不直接运行。Document 类已定义。")
    # 示例 Document 对象
    # doc_example = Document("这是一个示例文档。", metadata={"source": "示例来源"})
    # logger.info(f"示例文档: {doc_example}")
    pass

# src/rag/base_retriever.py
# RAG 系统中文档检索器的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional # Union 未在此文件中直接使用，但保持以防未来扩展

# 一个简单的文档数据结构。你可能想使用 Langchain/LlamaIndex 中更复杂的结构，
# 或者定义一个包含更多元数据的自定义结构。
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self.page_content = page_content # 页面内容
        self.metadata = metadata or {}  # 元数据
        # 允许将任意附加字段作为元数据的一部分，以提高灵活性
        if kwargs:
            self.metadata.update(kwargs)


    def __repr__(self):
        return f"Document(页面内容='{self.page_content[:50]}...', 元数据={self.metadata})"

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """
    RAG 系统中文档检索器的抽象基类。
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        为给定查询检索相关文档。

        参数:
            query (str): 要搜索的查询字符串。
            top_k (int): 要返回的相关性最高的文档数量。
            **kwargs: 用于检索的其他特定于提供程序的参数。

        返回:
            List[Document]: Document 对象列表。
        """
        pass

    async def aretrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        异步为给定查询检索相关文档。
        如果子类支持异步执行，则应重写此方法。
        """
        logger.warning(
            f"检索器 '{self.__class__.__name__}' 未实现特定的异步版本。"
            "将回退到同步检索。"
        )
        # 默认情况下，回退到同步执行。
        # 对于真正的异步，此方法需要是 `async def` 并使用 `await`。
        # 考虑使用 `asyncio.to_thread` 在异步上下文中运行同步代码。
        return self.retrieve(query, top_k, **kwargs)

    # 可选: 用于将文档添加到检索器的索引/存储库的方法
    def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        将文档添加到检索器的底层向量存储或索引中。
        并非所有检索器都支持直接添加文档 (有些可能是只读的)。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持直接添加文档。")

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        异步将文档添加到检索器的底层向量存储或索引中。
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持异步直接添加文档。")


if __name__ == '__main__':
    # 具体实现如何进行测试的示例 (概念性)
    class SimpleTestRetriever(BaseRetriever):
        def __init__(self, indexed_docs: Optional[List[Document]] = None):
            self.indexed_docs = indexed_docs or []
            if not self.indexed_docs: # 如果未提供，则添加一些默认文档
                 self.indexed_docs.extend([
                    Document(page_content="天空是蓝色的。", metadata={"source": "自然常识.txt", "id": "doc1"}),
                    Document(page_content="一天一苹果，医生远离我。", metadata={"source": "俗语.txt", "id": "doc2"}),
                    Document(page_content="Python 是一种通用的编程语言。", metadata={"source": "编程知识.md", "id": "doc3"}),
                    Document(page_content="蓝色的苹果很罕见。", metadata={"source": "自然常识.txt", "id": "doc4"}),
                ])


        def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
            logger.info(f"SimpleTestRetriever: 正在为查询 '{query}' 检索, top_k={top_k}")
            # 非常朴素的检索：返回包含查询中任何单词的文档
            query_words = set(query.lower().split())

            # 根据匹配单词的数量为文档评分 (非常基础)
            scored_docs = []
            for doc in self.indexed_docs:
                doc_words = set(doc.page_content.lower().split())
                common_words = query_words.intersection(doc_words)
                if common_words:
                    # 分配一个分数 (例如，共同词的数量)
                    # 添加对文档本身的引用
                    scored_docs.append({"score": len(common_words), "doc": doc})

            # 按分数降序排序
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            # 返回得分最高的 top_k 个文档的文档部分
            return [item["doc"] for item in scored_docs[:top_k]]

        def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
            self.indexed_docs.extend(documents)
            logger.info(f"SimpleTestRetriever: 已添加 {len(documents)} 个文档。总计: {len(self.indexed_docs)}")


    # 测试 SimpleTestRetriever
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    logger.info("使用 SimpleTestRetriever 实现测试 BaseRetriever...")
    retriever = SimpleTestRetriever()

    # 测试添加文档
    new_docs = [
        Document(page_content="敏捷的棕色狐狸跳过了懒狗。", metadata={"source": "字母全览句.txt", "id": "doc5"}),
        Document(page_content="大语言模型功能强大。", metadata={"source": "ai技术.txt", "id": "doc6"})
    ]
    retriever.add_documents(new_docs)
    assert len(retriever.indexed_docs) == 6

    # 测试检索
    query1 = "蓝色的东西"
    results1 = retriever.retrieve(query1, top_k=2)
    logger.info(f"查询 '{query1}' 的结果: {results1}")
    assert len(results1) <= 2
    if results1: # 确保结果是 Document 对象并且包含相关术语
        assert isinstance(results1[0], Document)
        assert "蓝色" in results1[0].page_content.lower() # "blue" -> "蓝色"


    query2 = "python 语言"
    results2 = retriever.retrieve(query2, top_k=1)
    logger.info(f"查询 '{query2}' 的结果: {results2}")
    assert len(results2) <= 1
    if results2:
        assert "python" in results2[0].page_content.lower()
        assert "语言" in results2[0].page_content.lower()

    query3 = "不存在的词语xyz"
    results3 = retriever.retrieve(query3)
    logger.info(f"查询 '{query3}' 的结果: {results3}")
    assert len(results3) == 0

    # 测试异步检索 (在此简单测试中回退到同步)
    import asyncio
    async def run_async_retrieve():
        logger.info("测试异步检索...")
        async_results = await retriever.aretrieve(query="苹果 医生", top_k=1)
        logger.info(f"查询 '苹果 医生' 的异步结果: {async_results}")
        assert len(async_results) <= 1
        if async_results:
            assert "苹果" in async_results[0].page_content.lower()
            assert "医生" in async_results[0].page_content.lower()

    asyncio.run(run_async_retrieve())

    logger.info("BaseRetriever 概念性测试完成。")

# src/rag/base_retriever.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

# A simple Document data structure. You might want to use a more complex one from Langchain/LlamaIndex
# or define your own with more metadata.
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self.page_content = page_content
        self.metadata = metadata or {}
        # Allow arbitrary additional fields to be part of metadata for flexibility
        if kwargs:
            self.metadata.update(kwargs)


    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers in a RAG system.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top relevant documents to return.
            **kwargs: Additional provider-specific arguments for retrieval.

        Returns:
            List[Document]: A list of Document objects.
        """
        pass

    async def aretrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
        """
        Asynchronously retrieves relevant documents for a given query.
        Subclasses should override this if they support asynchronous execution.
        """
        logger.warning(
            f"Retriever '{self.__class__.__name__}' does not have a specific async version implemented. "
            "Falling back to synchronous retrieve."
        )
        # By default, fall back to synchronous execution.
        # For true async, this needs to be `async def` and use `await`.
        # Consider using `asyncio.to_thread` for running sync code in async context.
        return self.retrieve(query, top_k, **kwargs)

    # Optional: Methods for adding documents to the retriever's index/store
    def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        Adds documents to the retriever's underlying vector store or index.
        Not all retrievers will support adding documents directly (some might be read-only).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support adding documents directly.")

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> None:
        """
        Asynchronously adds documents to the retriever's underlying vector store or index.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support async adding documents directly.")


if __name__ == '__main__':
    # Example of how a concrete implementation might be tested (conceptual)
    class SimpleTestRetriever(BaseRetriever):
        def __init__(self, indexed_docs: Optional[List[Document]] = None):
            self.indexed_docs = indexed_docs or []
            if not self.indexed_docs: # Add some default docs if none provided
                 self.indexed_docs.extend([
                    Document(page_content="The sky is blue.", metadata={"source": "nature_facts.txt", "id": "doc1"}),
                    Document(page_content="An apple a day keeps the doctor away.", metadata={"source": "sayings.txt", "id": "doc2"}),
                    Document(page_content="Python is a versatile programming language.", metadata={"source": "programming.md", "id": "doc3"}),
                    Document(page_content="Blue apples are rare.", metadata={"source": "nature_facts.txt", "id": "doc4"}),
                ])


        def retrieve(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Document]:
            logger.info(f"SimpleTestRetriever: Retrieving for query '{query}', top_k={top_k}")
            # Very naive retrieval: returns documents containing any word from the query
            query_words = set(query.lower().split())

            # Score documents based on number of matching words (very basic)
            scored_docs = []
            for doc in self.indexed_docs:
                doc_words = set(doc.page_content.lower().split())
                common_words = query_words.intersection(doc_words)
                if common_words:
                    # Assign a score (e.g., number of common words)
                    # Add a reference to the document itself
                    scored_docs.append({"score": len(common_words), "doc": doc})

            # Sort by score descending
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            # Return the document part of the top_k scored documents
            return [item["doc"] for item in scored_docs[:top_k]]

        def add_documents(self, documents: List[Document], **kwargs: Any) -> None:
            self.indexed_docs.extend(documents)
            logger.info(f"SimpleTestRetriever: Added {len(documents)} documents. Total: {len(self.indexed_docs)}")


    # Test the SimpleTestRetriever
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    logger.info("Testing BaseRetriever with SimpleTestRetriever implementation...")
    retriever = SimpleTestRetriever()

    # Test adding documents
    new_docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "pangrams.txt", "id": "doc5"}),
        Document(page_content="Large language models are powerful.", metadata={"source": "ai.txt", "id": "doc6"})
    ]
    retriever.add_documents(new_docs)
    assert len(retriever.indexed_docs) == 6

    # Test retrieval
    query1 = "blue things"
    results1 = retriever.retrieve(query1, top_k=2)
    logger.info(f"Results for '{query1}': {results1}")
    assert len(results1) <= 2
    if results1: # Ensure results are Documents and contain relevant terms
        assert isinstance(results1[0], Document)
        assert "blue" in results1[0].page_content.lower()


    query2 = "python language"
    results2 = retriever.retrieve(query2, top_k=1)
    logger.info(f"Results for '{query2}': {results2}")
    assert len(results2) <= 1
    if results2:
        assert "python" in results2[0].page_content.lower()
        assert "language" in results2[0].page_content.lower()

    query3 = "non_existent_term_xyz"
    results3 = retriever.retrieve(query3)
    logger.info(f"Results for '{query3}': {results3}")
    assert len(results3) == 0

    # Test async retrieval (falls back to sync in this simple test)
    import asyncio
    async def run_async_retrieve():
        logger.info("Testing async retrieve...")
        async_results = await retriever.aretrieve(query="apple doctor", top_k=1)
        logger.info(f"Async results for 'apple doctor': {async_results}")
        assert len(async_results) <= 1
        if async_results:
            assert "apple" in async_results[0].page_content.lower()
            assert "doctor" in async_results[0].page_content.lower()

    asyncio.run(run_async_retrieve())

    logger.info("BaseRetriever conceptual test finished.")

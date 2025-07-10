# src/rag/simple_rag.py
import logging
from typing import List, Dict, Any, Optional

from .base_retriever import BaseRetriever, Document # Assuming Document class is in base_retriever
from src.llms.base_llm import BaseLLM
from src.prompts.prompt_manager import PromptManager, DEFAULT_TEMPLATES_DIR

logger = logging.getLogger(__name__)

# Default RAG prompt template (can be overridden by creating this file)
# You would typically create a file like `rag_qa_prompt.txt` in your templates directory
# e.g., src/prompts/templates/rag_qa_prompt.txt
DEFAULT_RAG_PROMPT_NAME = "rag_qa_prompt"
DEFAULT_RAG_PROMPT_CONTENT = """\
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
$context_str

Question: $query

Answer:
"""

class SimpleRAG:
    """
    A simple Retrieval Augmented Generation pipeline.
    It uses a retriever to fetch documents and an LLM to generate an answer
    based on the query and retrieved context.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_manager: Optional[PromptManager] = None,
        rag_prompt_name: str = DEFAULT_RAG_PROMPT_NAME
    ):
        """
        Initializes the SimpleRAG pipeline.

        Args:
            retriever (BaseRetriever): The document retriever.
            llm (BaseLLM): The language model for generation.
            prompt_manager (Optional[PromptManager]): Manager for prompt templates.
                                                     If None, a default one is created.
            rag_prompt_name (str): The name of the RAG prompt template to use from PromptManager.
        """
        self.retriever = retriever
        self.llm = llm
        self.rag_prompt_name = rag_prompt_name

        if prompt_manager:
            self.prompt_manager = prompt_manager
        else:
            # Create a default prompt manager
            # Also ensure the default RAG prompt exists if not provided by user's templates
            self.prompt_manager = PromptManager() # Uses default templates dir
            if not self.prompt_manager.get_template(self.rag_prompt_name):
                logger.info(f"Default RAG prompt '{self.rag_prompt_name}' not found in PromptManager. "
                            f"Using built-in default content for it.")
                # Add the default content if the template file doesn't exist
                self.prompt_manager.loaded_templates[self.rag_prompt_name] = \
                    self.prompt_manager.PromptTemplate(DEFAULT_RAG_PROMPT_CONTENT)


        logger.info(f"SimpleRAG initialized with retriever: {retriever.__class__.__name__}, "
                    f"LLM: {llm.model_name}, RAG prompt: '{self.rag_prompt_name}'")


    def _format_context(self, documents: List[Document]) -> str:
        """
        Formats a list of documents into a single string for the context.
        """
        if not documents:
            return "No context available."
        # Simple formatting: join page content with newlines
        return "\n\n---\n\n".join([doc.page_content for doc in documents])


    def answer_query(
        self,
        query: str,
        top_k_retrieval: int = 3,
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        **kwargs: Any # Additional args for retriever or LLM
    ) -> Dict[str, Any]:
        """
        Answers a query using the RAG pipeline.

        Args:
            query (str): The user's query.
            top_k_retrieval (int): Number of documents to retrieve.
            llm_max_tokens (Optional[int]): Max tokens for LLM generation.
            llm_temperature (Optional[float]): Temperature for LLM generation.
            **kwargs: Additional arguments that might be passed to retriever's retrieve method
                      or LLM's generate/chat method.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "answer": The generated answer string.
                - "retrieved_documents": A list of Document objects used as context.
                - "formatted_prompt": The full prompt sent to the LLM.
        """
        logger.info(f"RAG answering query: '{query}', top_k={top_k_retrieval}")

        # 1. Retrieve documents
        retriever_args = kwargs.get("retriever_args", {})
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieval, **retriever_args)
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query '{query}'.")

        # 2. Format context
        context_str = self._format_context(retrieved_docs)

        # 3. Prepare prompt
        prompt_vars = {
            "context_str": context_str,
            "query": query,
            **kwargs.get("prompt_vars", {}) # Allow overriding/adding prompt vars
        }
        formatted_prompt = self.prompt_manager.format_prompt(self.rag_prompt_name, **prompt_vars)

        if not formatted_prompt:
            logger.error(f"Could not format RAG prompt '{self.rag_prompt_name}'. Aborting.")
            return {
                "answer": "Error: Could not generate an answer due to prompt formatting issues.",
                "retrieved_documents": retrieved_docs,
                "formatted_prompt": None
            }

        logger.debug(f"Formatted RAG prompt:\n{formatted_prompt}")

        # 4. Generate answer with LLM
        llm_args = kwargs.get("llm_args", {})
        # Decide whether to use chat or generate based on LLM capabilities or model name
        # This is a heuristic; a more robust way might be to check for a 'chat' method on the llm instance.
        if "chat" in dir(self.llm) and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            # Assuming the prompt is a user message for a chat model
            messages = [
                # If your RAG prompt template is designed as a system prompt, use that here.
                # Or, if it's a full user instruction:
                {"role": "user", "content": formatted_prompt}
            ]
            llm_response = self.llm.chat(
                messages,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )
            answer = llm_response.get("content", "Error: No content in LLM chat response.")
        else:
            answer = self.llm.generate(
                formatted_prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )

        logger.info(f"RAG generated answer for '{query}': '{answer[:100]}...'")

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "formatted_prompt": formatted_prompt
        }

    async def aanswer_query(
        self,
        query: str,
        top_k_retrieval: int = 3,
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Asynchronously answers a query using the RAG pipeline.
        """
        logger.info(f"RAG async answering query: '{query}', top_k={top_k_retrieval}")

        retriever_args = kwargs.get("retriever_args", {})
        retrieved_docs = await self.retriever.aretrieve(query, top_k=top_k_retrieval, **retriever_args)
        logger.debug(f"Async retrieved {len(retrieved_docs)} documents for query '{query}'.")

        context_str = self._format_context(retrieved_docs)

        prompt_vars = {
            "context_str": context_str,
            "query": query,
            **kwargs.get("prompt_vars", {})
        }
        formatted_prompt = self.prompt_manager.format_prompt(self.rag_prompt_name, **prompt_vars)

        if not formatted_prompt:
            logger.error(f"Async: Could not format RAG prompt '{self.rag_prompt_name}'. Aborting.")
            return {
                "answer": "Error: Could not generate an answer due to prompt formatting issues.",
                "retrieved_documents": retrieved_docs,
                "formatted_prompt": None
            }
        logger.debug(f"Async formatted RAG prompt:\n{formatted_prompt}")

        llm_args = kwargs.get("llm_args", {})
        if "achat" in dir(self.llm) and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            messages = [{"role": "user", "content": formatted_prompt}]
            llm_response = await self.llm.achat(
                messages,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )
            answer = llm_response.get("content", "Error: No content in async LLM chat response.")
        else:
            answer = await self.llm.agenerate(
                formatted_prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )

        logger.info(f"Async RAG generated answer for '{query}': '{answer[:100]}...'")

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "formatted_prompt": formatted_prompt
        }


if __name__ == '__main__':
    # This test requires a concrete LLM, Retriever, and PromptManager setup.
    # For simplicity, we'll use mock/simple versions.
    import asyncio
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM # Using OpenAI for a concrete test
    from src.rag.base_retriever import SimpleTestRetriever # The test retriever from base_retriever.py

    load_config()
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY not set or is a placeholder. Skipping SimpleRAG integration test.")
    else:
        logger.info("Testing SimpleRAG pipeline...")

        # 1. Setup components
        mock_llm = OpenAILLM(model_name="gpt-3.5-turbo") # Requires OPENAI_API_KEY

        # Use the SimpleTestRetriever with its default docs
        # Docs: "The sky is blue.", "An apple a day keeps the doctor away.", "Python is a versatile programming language.", "Blue apples are rare."
        mock_retriever = SimpleTestRetriever()

        # Prompt Manager (will use default RAG prompt if rag_qa_prompt.txt doesn't exist)
        # Let's ensure our test RAG prompt is available in the prompt manager for this test
        pm = PromptManager()
        test_rag_prompt_name = "test_rag_qa_prompt"
        test_rag_prompt_content = "Context: $context_str\nQuestion: $query\nAnswer based ONLY on context:"
        if not pm.get_template(test_rag_prompt_name):
             pm.loaded_templates[test_rag_prompt_name] = pm.PromptTemplate(test_rag_prompt_content)


        # 2. Initialize RAG
        rag_pipeline = SimpleRAG(retriever=mock_retriever, llm=mock_llm, prompt_manager=pm, rag_prompt_name=test_rag_prompt_name)

        # 3. Test query
        test_query = "What color is the sky?"
        logger.info(f"\n--- Testing RAG with query: '{test_query}' ---")
        result = rag_pipeline.answer_query(test_query, top_k_retrieval=2)

        logger.info(f"RAG Answer: {result['answer']}")
        logger.info(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for doc in result['retrieved_documents']:
            logger.info(f"  - {doc.page_content} (Source: {doc.metadata.get('source')})")
        # logger.info(f"Full prompt to LLM:\n{result['formatted_prompt']}")

        assert "blue" in result['answer'].lower() # Expecting the LLM to pick this up
        assert any("sky is blue" in doc.page_content for doc in result['retrieved_documents'])


        test_query_python = "Tell me about Python."
        logger.info(f"\n--- Testing RAG with query: '{test_query_python}' ---")
        result_python = rag_pipeline.answer_query(test_query_python, top_k_retrieval=1)
        logger.info(f"RAG Answer: {result_python['answer']}")
        assert "python" in result_python['answer'].lower()
        assert any("Python is a versatile programming language" in doc.page_content for doc in result_python['retrieved_documents'])

        # Test async query
        async def run_async_rag_test():
            logger.info(f"\n--- Testing Async RAG with query: '{test_query}' ---")
            async_result = await rag_pipeline.aanswer_query(test_query, top_k_retrieval=2)
            logger.info(f"Async RAG Answer: {async_result['answer']}")
            logger.info(f"Async Retrieved Documents ({len(async_result['retrieved_documents'])}):")
            for doc in async_result['retrieved_documents']:
                logger.info(f"  - {doc.page_content} (Source: {doc.metadata.get('source')})")
            assert "blue" in async_result['answer'].lower()

        asyncio.run(run_async_rag_test())

        logger.info("SimpleRAG tests completed.")

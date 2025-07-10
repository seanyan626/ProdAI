# src/rag/simple_rag.py
# 简单的 RAG（检索增强生成）流程实现
import logging
from typing import List, Dict, Any, Optional

from .base_retriever import BaseRetriever, Document # 假设 Document 类在 base_retriever 中
from src.llms.base_llm import BaseLLM
from src.prompts.prompt_manager import PromptManager # DEFAULT_TEMPLATES_DIR 未在此处直接使用

logger = logging.getLogger(__name__)

# 默认 RAG 提示模板 (可以通过创建此文件来覆盖)
# 你通常会在模板目录中创建一个类似 `rag_qa_prompt.txt` 的文件
# 例如 src/prompts/templates/rag_qa_prompt.txt
DEFAULT_RAG_PROMPT_NAME = "rag_qa_prompt" # 默认RAG问答提示名称
DEFAULT_RAG_PROMPT_CONTENT = """\
你是一个用于问答任务的助手。
请使用以下检索到的上下文片段来回答问题。
如果你不知道答案，就直接说你不知道。
最多使用三个句子，并保持答案简洁。

上下文:
$context_str

问题: $query

答案:
"""

class SimpleRAG:
    """
    一个简单的检索增强生成 (RAG) 流程。
    它使用检索器获取文档，并使用 LLM 根据查询和检索到的上下文生成答案。
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        prompt_manager: Optional[PromptManager] = None,
        rag_prompt_name: str = DEFAULT_RAG_PROMPT_NAME
    ):
        """
        初始化 SimpleRAG 流程。

        参数:
            retriever (BaseRetriever): 文档检索器。
            llm (BaseLLM): 用于生成的语言模型。
            prompt_manager (Optional[PromptManager]): 提示模板管理器。
                                                     如果为 None，则创建一个默认的。
            rag_prompt_name (str): 要从 PromptManager 使用的 RAG 提示模板的名称。
        """
        self.retriever = retriever
        self.llm = llm
        self.rag_prompt_name = rag_prompt_name

        if prompt_manager:
            self.prompt_manager = prompt_manager
        else:
            # 创建一个默认的提示管理器
            # 同时确保如果用户的模板中未提供默认 RAG 提示，则该提示存在
            self.prompt_manager = PromptManager() # 使用默认模板目录
            if not self.prompt_manager.get_template(self.rag_prompt_name):
                logger.info(f"在 PromptManager 中未找到默认 RAG 提示 '{self.rag_prompt_name}'。"
                            f"将为其使用内置的默认内容。")
                # 如果模板文件不存在，则添加默认内容
                self.prompt_manager.loaded_templates[self.rag_prompt_name] = \
                    self.prompt_manager.PromptTemplate(DEFAULT_RAG_PROMPT_CONTENT)


        logger.info(f"SimpleRAG 已初始化，检索器: {retriever.__class__.__name__}, "
                    f"LLM: {llm.model_name}, RAG 提示: '{self.rag_prompt_name}'")


    def _format_context(self, documents: List[Document]) -> str:
        """
        将文档列表格式化为单个字符串作为上下文。
        """
        if not documents:
            return "没有可用的上下文。"
        # 简单格式化：用换行符连接页面内容
        return "\n\n---\n\n".join([doc.page_content for doc in documents])


    def answer_query(
        self,
        query: str,
        top_k_retrieval: int = 3,
        llm_max_tokens: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        **kwargs: Any # 用于检索器或 LLM 的其他参数
    ) -> Dict[str, Any]:
        """
        使用 RAG 流程回答查询。

        参数:
            query (str): 用户的查询。
            top_k_retrieval (int): 要检索的文档数量。
            llm_max_tokens (Optional[int]): LLM 生成的最大 token 数。
            llm_temperature (Optional[float]): LLM 生成的温度。
            **kwargs: 可能传递给检索器的 retrieve 方法或 LLM 的 generate/chat 方法的其他参数。

        返回:
            Dict[str, Any]: 包含以下内容的字典：
                - "answer": 生成的答案字符串。
                - "retrieved_documents": 用作上下文的 Document 对象列表。
                - "formatted_prompt": 发送给 LLM 的完整提示。
        """
        logger.info(f"RAG 正在回答查询: '{query}', top_k={top_k_retrieval}")

        # 1. 检索文档
        retriever_args = kwargs.get("retriever_args", {})
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k_retrieval, **retriever_args)
        logger.debug(f"为查询 '{query}' 检索到 {len(retrieved_docs)} 个文档。")

        # 2. 格式化上下文
        context_str = self._format_context(retrieved_docs)

        # 3. 准备提示
        prompt_vars = {
            "context_str": context_str,
            "query": query,
            **kwargs.get("prompt_vars", {}) # 允许覆盖/添加提示变量
        }
        formatted_prompt = self.prompt_manager.format_prompt(self.rag_prompt_name, **prompt_vars)

        if not formatted_prompt:
            logger.error(f"无法格式化 RAG 提示 '{self.rag_prompt_name}'。正在中止。")
            return {
                "answer": "错误: 由于提示格式化问题，无法生成答案。",
                "retrieved_documents": retrieved_docs,
                "formatted_prompt": None
            }

        logger.debug(f"格式化的 RAG 提示:\n{formatted_prompt}")

        # 4. 使用 LLM 生成答案
        llm_args = kwargs.get("llm_args", {})
        # 根据 LLM 的能力或模型名称决定是使用 chat 还是 generate
        # 这是一种启发式方法；更健壮的方法可能是检查 llm 实例上是否存在 'chat' 方法。
        if "chat" in dir(self.llm) and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            # 假设提示是聊天模型的用户消息
            messages = [
                # 如果你的 RAG 提示模板被设计为系统提示，请在此处使用它。
                # 或者，如果它是一个完整的用户指令：
                {"role": "user", "content": formatted_prompt}
            ]
            llm_response = self.llm.chat(
                messages,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )
            answer = llm_response.get("content", "错误: LLM 聊天响应中无内容。")
        else:
            answer = self.llm.generate(
                formatted_prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )

        logger.info(f"RAG 为 '{query}' 生成的答案: '{answer[:100]}...'")

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
        异步使用 RAG 流程回答查询。
        """
        logger.info(f"RAG 正在异步回答查询: '{query}', top_k={top_k_retrieval}")

        retriever_args = kwargs.get("retriever_args", {})
        retrieved_docs = await self.retriever.aretrieve(query, top_k=top_k_retrieval, **retriever_args)
        logger.debug(f"为查询 '{query}' 异步检索到 {len(retrieved_docs)} 个文档。")

        context_str = self._format_context(retrieved_docs)

        prompt_vars = {
            "context_str": context_str,
            "query": query,
            **kwargs.get("prompt_vars", {})
        }
        formatted_prompt = self.prompt_manager.format_prompt(self.rag_prompt_name, **prompt_vars)

        if not formatted_prompt:
            logger.error(f"异步: 无法格式化 RAG 提示 '{self.rag_prompt_name}'。正在中止。")
            return {
                "answer": "错误: 由于提示格式化问题，无法生成答案。",
                "retrieved_documents": retrieved_docs,
                "formatted_prompt": None
            }
        logger.debug(f"异步格式化的 RAG 提示:\n{formatted_prompt}")

        llm_args = kwargs.get("llm_args", {})
        if "achat" in dir(self.llm) and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            messages = [{"role": "user", "content": formatted_prompt}]
            llm_response = await self.llm.achat(
                messages,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )
            answer = llm_response.get("content", "错误: 异步 LLM 聊天响应中无内容。")
        else:
            answer = await self.llm.agenerate(
                formatted_prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                **llm_args
            )

        logger.info(f"异步 RAG 为 '{query}' 生成的答案: '{answer[:100]}...'")

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "formatted_prompt": formatted_prompt
        }


if __name__ == '__main__':
    # 此测试需要具体的 LLM、Retriever 和 PromptManager 设置。
    # 为简单起见，我们将使用模拟/简单版本。
    import asyncio
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM # 使用 OpenAI 进行具体测试
    from src.rag.base_retriever import SimpleTestRetriever # 来自 base_retriever.py 的测试检索器

    load_config()
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY 未设置或为占位符。正在跳过 SimpleRAG 集成测试。")
    else:
        logger.info("正在测试 SimpleRAG 流程...")

        # 1. 设置组件
        mock_llm = OpenAILLM(model_name="gpt-3.5-turbo") # 需要 OPENAI_API_KEY

        # 使用 SimpleTestRetriever 及其默认文档
        # 文档: "天空是蓝色的。", "一天一苹果，医生远离我。", "Python 是一种通用的编程语言。", "蓝色的苹果很罕见。"
        # (注意：SimpleTestRetriever 中的示例文档内容已在本地化 base_retriever.py 时更新为中文)
        mock_retriever = SimpleTestRetriever()

        # 提示管理器 (如果 rag_qa_prompt.txt 不存在，将使用默认 RAG 提示)
        # 让我们确保测试 RAG 提示在此测试的提示管理器中可用
        pm = PromptManager()
        test_rag_prompt_name = "test_rag_qa_prompt_测试用" # 测试RAG问答提示名称
        test_rag_prompt_content = "上下文: $context_str\n问题: $query\n请仅根据上下文回答:"
        if not pm.get_template(test_rag_prompt_name):
             pm.loaded_templates[test_rag_prompt_name] = pm.PromptTemplate(test_rag_prompt_content)


        # 2. 初始化 RAG
        rag_pipeline = SimpleRAG(retriever=mock_retriever, llm=mock_llm, prompt_manager=pm, rag_prompt_name=test_rag_prompt_name)

        # 3. 测试查询
        test_query = "天空是什么颜色的？"
        logger.info(f"\n--- 使用查询测试 RAG: '{test_query}' ---")
        result = rag_pipeline.answer_query(test_query, top_k_retrieval=2)

        logger.info(f"RAG 答案: {result['answer']}")
        logger.info(f"检索到的文档 ({len(result['retrieved_documents'])}):")
        for doc in result['retrieved_documents']:
            logger.info(f"  - {doc.page_content} (来源: {doc.metadata.get('source')})")
        # logger.info(f"发送给 LLM 的完整提示:\n{result['formatted_prompt']}")

        assert "蓝" in result['answer'].lower() # 期望 LLM 能捕捉到这个 (blue -> 蓝)
        assert any("天空是蓝色的" in doc.page_content for doc in result['retrieved_documents'])


        test_query_python = "告诉我关于 Python 的信息。"
        logger.info(f"\n--- 使用查询测试 RAG: '{test_query_python}' ---")
        result_python = rag_pipeline.answer_query(test_query_python, top_k_retrieval=1)
        logger.info(f"RAG 答案: {result_python['answer']}")
        assert "python" in result_python['answer'].lower() # Python 通常保持英文
        assert any("Python 是一种通用的编程语言" in doc.page_content for doc in result_python['retrieved_documents'])

        # 测试异步查询
        async def run_async_rag_test():
            logger.info(f"\n--- 使用查询测试异步 RAG: '{test_query}' ---")
            async_result = await rag_pipeline.aanswer_query(test_query, top_k_retrieval=2)
            logger.info(f"异步 RAG 答案: {async_result['answer']}")
            logger.info(f"异步检索到的文档 ({len(async_result['retrieved_documents'])}):")
            for doc in async_result['retrieved_documents']:
                logger.info(f"  - {doc.page_content} (来源: {doc.metadata.get('source')})")
            assert "蓝" in async_result['answer'].lower()

        asyncio.run(run_async_rag_test())

        logger.info("SimpleRAG 测试完成。")

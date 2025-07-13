# core/agents/specific_agent.py
# 一个更具体的 Agent 示例实现
import logging
from typing import Any, List, Dict, Optional, Union

from core.models.llm.base_llm import BaseLLM  # 更新路径和类名
from core.prompts.prompt_manager import PromptManager
from core.tools.base_tool import BaseTool
from .base_agent import BaseAgent, AgentAction, AgentFinish

logger = logging.getLogger(__name__)

DEFAULT_SPECIFIC_AGENT_PROMPT_NAME = "react_parser_agent_prompt"
DEFAULT_SPECIFIC_AGENT_PROMPT_CONTENT = """
你是一个乐于助人的助手，可以使用工具来回答问题。
你的目标是准确简洁地回答用户的问题。

工具 (TOOLS):
你可以使用以下工具:
$tool_descriptions

要使用工具，你必须使用以下格式:
Thought: [你选择工具和行动的推理过程]
Action: [要使用的工具的确切名称，例如 web_search]
Action Input: [工具输入的 JSON 兼容字典字符串，需匹配其参数模式。例如 {"query": "当前天气"}]

行动之后，你会收到一个观察结果。
Observation: [工具返回的结果]

如果你认为根据观察结果和你的知识已经有足够的信息来回答问题，
你必须使用以下格式:
Thought: [你为什么现在可以回答的推理过程]
Final Answer: [你对原始用户问题的全面回答]

对话历史 (Conversation History):
$chat_history

用户问题 (User Question): $input

暂存区 (Scratchpad - 你的思考、行动和观察):
$scratchpad

Thought:
"""


class SpecificAgent(BaseAgent):
    """
    一个更具体的 Agent 示例，可能使用 ReAct 风格的提示机制
    来决定行动并解析 LLM 响应。
    （实现待补充）
    """

    def __init__(
            self,
            llm: BaseLLM,  # <--- 已更改回 BaseLLM
            tools: Optional[List[BaseTool]] = None,
            prompt_manager: Optional[PromptManager] = None,
            agent_prompt_name: str = DEFAULT_SPECIFIC_AGENT_PROMPT_NAME,
            **kwargs: Any
    ):
        resolved_pm = prompt_manager or PromptManager()

        if agent_prompt_name == DEFAULT_SPECIFIC_AGENT_PROMPT_NAME and \
                not resolved_pm.get_template(DEFAULT_SPECIFIC_AGENT_PROMPT_NAME):
            logger.info(f"未找到默认提示 '{DEFAULT_SPECIFIC_AGENT_PROMPT_NAME}'。正在使用内置内容。")
            resolved_pm.loaded_templates[DEFAULT_SPECIFIC_AGENT_PROMPT_NAME] = \
                resolved_pm.PromptTemplate(DEFAULT_SPECIFIC_AGENT_PROMPT_CONTENT)

        super().__init__(llm, tools, prompt_manager=resolved_pm, agent_prompt_name=agent_prompt_name, **kwargs)
        logger.info(f"SpecificAgent 已使用提示模板 '{self.agent_prompt_name}' 初始化。")

    def _parse_llm_output(self, llm_output: str) -> Union[AgentAction, AgentFinish, None]:
        """
        解析 LLM 的文本输出以查找行动或最终答案。
        （实现待补充）
        """
        # logger.debug(f"正在解析 LLM 输出:\n---\n{llm_output}\n---")
        # ... (原解析逻辑已移除)
        # 这是一个非常简化的模拟解析，实际应用中需要更复杂的逻辑
        if "Final Answer:" in llm_output:
            return AgentFinish(output={"answer": "来自LLM的模拟最终答案"}, log="模拟的思考过程")
        elif "Action:" in llm_output and "Action Input:" in llm_output:
            # 极简模拟，不真正解析工具名称和输入
            return AgentAction(tool_name="simulated_tool", tool_input={"query": "模拟查询"}, log="模拟的思考过程")
        return None

    def _plan(
            self,
            inputs: Dict[str, Any],
            intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        SpecificAgent 的规划逻辑。
        （实现待补充）
        """
        # current_user_input = inputs.get("input", "")
        # scratchpad_str = self._construct_scratchpad(intermediate_steps)
        # tool_info_str = self._get_tool_info_string()
        # chat_history_list = self.memory.get_history(max_messages=5)
        # chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])
        # prompt_vars = { ... }
        # full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)
        # if not full_prompt:
        #     # ... (错误处理)
        # logger.debug(f"--- SpecificAgent 发送给 LLM 的提示 ---\n{full_prompt}\n---")
        # llm_response_text = self.llm.generate(full_prompt, stop_sequences=["\nObservation:"]) # 或 chat
        # if not llm_response_text.strip():
        #     # ... (错误处理)
        # parsed_decision = self._parse_llm_output(llm_response_text)
        # if parsed_decision is None:
        #     # ... (错误处理)
        # return parsed_decision

        # 简化占位符实现：
        logger.info(f"SpecificAgent._plan 调用，输入: {inputs.get('input', '')[:50]}...")
        if not intermediate_steps:  # 如果是第一步，模拟一个动作
            return AgentAction(tool_name="search_tool", tool_input={"query": inputs.get("input", "")},
                               log="需要搜索信息")
        else:  # 否则，模拟完成
            return AgentFinish(output={"answer": "这是来自 SpecificAgent 的模拟答案。"}, log="已完成处理。")

    async def _aplan(
            self,
            inputs: Dict[str, Any],
            intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        SpecificAgent 的异步规划逻辑。
        （实现待补充）
        """
        logger.info(f"SpecificAgent._aplan 调用，输入: {inputs.get('input', '')[:50]}...")
        # 异步版本也使用类似的简化逻辑
        if not intermediate_steps:
            return AgentAction(tool_name="search_tool", tool_input={"query": inputs.get("input", "")},
                               log="需要异步搜索信息")
        else:
            return AgentFinish(output={"answer": "这是来自 SpecificAgent 的异步模拟答案。"}, log="已异步完成处理。")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging

    # from src.llms.openai_llm import OpenAILLM # 实际测试时需要
    # from src.tools.search_tool import SearchTool # 实际测试时需要
    # from src.memory.simple_memory import SimpleMemory # 实际测试时需要
    # import asyncio # 实际测试时需要

    load_config()
    setup_logging()
    logger.info("SpecificAgent 模块可以直接运行测试（如果包含测试代码和必要的模拟/真实组件）。")
    # 此处可以添加直接测试此模块内函数的代码
    pass

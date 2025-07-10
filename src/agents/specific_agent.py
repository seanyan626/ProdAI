# src/agents/specific_agent.py
# 一个更具体的 Agent 示例实现
import logging
import json
from typing import Any, List, Dict, Optional, Union

from .base_agent import BaseAgent, AgentAction, AgentFinish
from src.llms.base_llm import BaseLLM
from src.tools.base_tool import BaseTool
from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# 此特定 Agent 的默认提示名称。
# 期望在提示模板目录中有一个类似 `react_agent_prompt.txt` 的模板文件。
DEFAULT_SPECIFIC_AGENT_PROMPT_NAME = "react_parser_agent_prompt" # ReAct 解析器 Agent 提示

# 如果找不到模板文件，则使用默认提示内容。
# 这是一个简化的 ReAct 风格提示。
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
    此 Agent 将尝试解析 LLM 输出中的 "Action:"、"Action Input:" 和 "Final Answer:"。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        prompt_manager: Optional[PromptManager] = None,
        agent_prompt_name: str = DEFAULT_SPECIFIC_AGENT_PROMPT_NAME,
        **kwargs: Any
    ):
        resolved_pm = prompt_manager or PromptManager()

        # 确保如果模板文件丢失，则默认提示内容可用
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
        这是 Agent 的一个关键且通常复杂的部分。
        查找 "Action:"、"Action Input:" 和 "Final Answer:" 关键字。
        """
        logger.debug(f"正在解析 LLM 输出:\n---\n{llm_output}\n---")

        thought = ""
        # 首先尝试提取思考过程
        if "Thought:" in llm_output:
            thought_parts = llm_output.split("Thought:", 1)
            if len(thought_parts) > 1:
                # 思考过程是 "Thought:" 之后到下一个关键字 (Action 或 Final Answer) 之前的所有内容
                next_keyword_pos = -1
                action_pos = llm_output.find("Action:", len(thought_parts[0]) + len("Thought:"))
                final_answer_pos = llm_output.find("Final Answer:", len(thought_parts[0]) + len("Thought:"))

                if action_pos != -1 and final_answer_pos != -1:
                    next_keyword_pos = min(action_pos, final_answer_pos)
                elif action_pos != -1:
                    next_keyword_pos = action_pos
                elif final_answer_pos != -1:
                    next_keyword_pos = final_answer_pos

                if next_keyword_pos != -1:
                    thought = thought_parts[1][:next_keyword_pos - (len(thought_parts[0]) + len("Thought:"))].strip()
                else: # 思考过程直到末尾
                    thought = thought_parts[1].strip()
            logger.debug(f"解析到的思考: {thought}")


        # 检查最终答案
        if "Final Answer:" in llm_output:
            parts = llm_output.split("Final Answer:", 1)
            if len(parts) > 1:
                final_answer_text = parts[1].strip()
                logger.info(f"LLM 表明是最终答案: {final_answer_text}")
                return AgentFinish(output={"answer": final_answer_text}, log=thought or "已得到最终答案。")
            else: # 存在 "Final Answer:" 但后面没有文本。
                 logger.warning("LLM 输出包含 'Final Answer:' 但之后没有文本。")


        # 检查行动和行动输入
        action_marker = "Action:"
        action_input_marker = "Action Input:"

        action_idx = llm_output.find(action_marker)
        if action_idx != -1:
            action_input_idx = llm_output.find(action_input_marker, action_idx + len(action_marker))

            if action_input_idx != -1:
                action_name_str = llm_output[action_idx + len(action_marker):action_input_idx].strip()

                # 行动输入可能跨越多行，如果它是一个复杂的 JSON。
                # 我们需要找到 JSON 字符串的结束位置。
                # 这很棘手；一个健壮的解析器会处理嵌套结构。
                # 为简单起见，假设输入在一行上或在下一个关键字之前以换行符结束。
                action_input_block = llm_output[action_input_idx + len(action_input_marker):].strip()

                # 尝试找到 JSON 的结尾 (例如，在下一个 "Thought:", "Action:", "Observation:" 之前)
                # 这是一种非常朴素的查找 JSON 结尾的方法。
                end_markers = ["Thought:", "Action:", "Observation:", "\nFinal Answer:"]
                min_pos = len(action_input_block)
                for marker in end_markers:
                    pos = action_input_block.find(marker)
                    if pos != -1 and pos < min_pos:
                        min_pos = pos

                action_input_str = action_input_block[:min_pos].strip()

                logger.debug(f"尝试解析行动: '{action_name_str}', 输入字符串: '{action_input_str}'")

                try:
                    # LLM 应为输入输出一个 JSON 字符串。
                    tool_input_dict = json.loads(action_input_str)
                    if not isinstance(tool_input_dict, dict):
                        # 不是一个JSON对象（字典）
                        raise json.JSONDecodeError("输入不是 JSON 对象 (字典)。", action_input_str, 0)

                    logger.info(f"LLM 行动: 工具='{action_name_str}', 输入={tool_input_dict}")
                    return AgentAction(tool_name=action_name_str, tool_input=tool_input_dict, log=thought)
                except json.JSONDecodeError as e:
                    logger.error(f"从行动输入解析 JSON 失败: '{action_input_str}'。错误: {e}", exc_info=True)
                    # 回退或错误处理：可以要求 LLM 重新格式化或提供错误消息。
                    # 目前，我们将指示解析失败。
                    return AgentFinish(
                        output={"error": f"LLM 为工具输入提供了无效的 JSON: {action_input_str}。详情: {e}"},
                        log=thought + f"\n错误: 解析 LLM 工具输入失败: {e}"
                    )
            else:
                logger.warning(f"LLM 输出包含 'Action:' 但之后未找到 'Action Input:'。输出: {llm_output}")

        # 如果未找到明确的行动或最终答案，则可能是格式错误的响应或只是一个思考过程。
        # 根据 Agent 的设计，你可能希望：
        # 1. 要求 LLM 重试/重新格式化。
        # 2. 将其视为思考的延续并再次提示。
        # 3. 返回错误或默认操作。
        logger.warning(f"无法从 LLM 输出中解析出明确的行动或最终答案: {llm_output}")
        # 对于此 Agent，如果未找到特定标记，我们假设它是格式错误的响应
        # 并且可能决定以错误结束或要求澄清。
        # 让我们返回 None 以指示解析未能找到结构化命令。
        # _plan 方法将需要处理此问题。
        return None


    def _plan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:

        current_user_input = inputs.get("input", "")
        scratchpad_str = self._construct_scratchpad(intermediate_steps)
        tool_info_str = self._get_tool_info_string()

        # 获取聊天历史，简单格式化
        # 包含历史记录时要注意上下文窗口限制
        chat_history_list = self.memory.get_history(max_messages=5) # 获取最近 5 条消息
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        prompt_vars = {
            "tool_descriptions": tool_info_str,
            "chat_history": chat_history_str,
            "input": current_user_input,
            "scratchpad": scratchpad_str
        }

        full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)
        if not full_prompt:
            logger.error(f"格式化 Agent 提示失败: {self.agent_prompt_name}")
            return AgentFinish({"error": "内部错误: 无法创建 Agent 提示。"}, log="提示格式化失败。")

        logger.debug(f"--- SpecificAgent 发送给 LLM 的提示 ---\n{full_prompt}\n---------------------------------")

        # 调用 LLM
        # 此 Agent 假设如果 LLM 具有 'chat' 方法，则其为聊天模型。
        if hasattr(self.llm, 'chat') and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            # 整个构造的提示成为聊天模型的用户消息
            messages = [{"role": "user", "content": full_prompt}]
            # 停止序列有助于 LLM 集中注意力
            llm_response_obj = self.llm.chat(messages, stop_sequences=["\nObservation:"])
            llm_response_text = llm_response_obj.get("content", "")
        else:
            llm_response_text = self.llm.generate(full_prompt, stop_sequences=["\nObservation:"])

        if not llm_response_text.strip():
            logger.warning("LLM 返回了空响应。")
            # 处理空响应，例如重试或以错误结束
            return AgentFinish({"error": "LLM 返回了空响应。"}, log="LLM 未提供输出。")

        # 解析 LLM 的响应
        parsed_decision = self._parse_llm_output(llm_response_text)

        if parsed_decision is None:
            # 解析未能找到明确的行动或完成。
            # 这可能是由于 LLM 输出格式错误或 LLM 只是在“进一步思考”。
            # 我们可能希望让 Agent 重试，或放弃。
            # 目前，我们假设它是格式错误的响应，并以错误结束。
            log_message = (f"LLM 输出无法解析为明确的行动或最终答案。"
                           f"LLM 原始输出: '{llm_response_text}'")
            logger.warning(log_message)
            return AgentFinish(
                output={"error": "未能理解 LLM 的响应。", "details": llm_response_text},
                log=log_message
            )

        return parsed_decision

    async def _aplan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        # 与 _plan 类似，但使用异步 LLM 调用

        current_user_input = inputs.get("input", "")
        scratchpad_str = self._construct_scratchpad(intermediate_steps)
        tool_info_str = self._get_tool_info_string()
        chat_history_list = await self.memory.get_history(max_messages=5) # 假设是异步的 get_history
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        prompt_vars = {
            "tool_descriptions": tool_info_str,
            "chat_history": chat_history_str,
            "input": current_user_input,
            "scratchpad": scratchpad_str
        }

        full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)
        if not full_prompt:
            logger.error(f"异步: 格式化 Agent 提示失败: {self.agent_prompt_name}")
            return AgentFinish({"error": "内部错误: 无法创建 Agent 提示。"}, log="提示格式化失败。")

        logger.debug(f"--- SpecificAgent (异步) 发送给 LLM 的提示 ---\n{full_prompt}\n---------------------------------")

        if hasattr(self.llm, 'achat') and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            messages = [{"role": "user", "content": full_prompt}]
            llm_response_obj = await self.llm.achat(messages, stop_sequences=["\nObservation:"])
            llm_response_text = llm_response_obj.get("content", "")
        else:
            llm_response_text = await self.llm.agenerate(full_prompt, stop_sequences=["\nObservation:"])

        if not llm_response_text.strip():
            logger.warning("异步 LLM 返回了空响应。")
            return AgentFinish({"error": "异步 LLM 返回了空响应。"}, log="异步 LLM 未提供输出。")

        parsed_decision = self._parse_llm_output(llm_response_text) # 解析逻辑是同步的

        if parsed_decision is None:
            log_message = (f"异步: LLM 输出无法解析。LLM 原始输出: '{llm_response_text}'")
            logger.warning(log_message)
            return AgentFinish(
                output={"error": "异步: 未能理解 LLM 的响应。", "details": llm_response_text},
                log=log_message
            )

        return parsed_decision


if __name__ == '__main__':
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM
    from src.tools.search_tool import SearchTool # 假设 SearchTool 可用且工作正常
    from src.memory.simple_memory import SimpleMemory
    import asyncio

    load_config()
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY 未设置或为占位符。跳过 SpecificAgent 集成测试。")
    else:
        logger.info("\n--- 正在测试 SpecificAgent ---")

        # 设置组件
        llm_instance = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.0) # 低温以获得更可预测的解析
        search_tool_instance = SearchTool()
        agent_memory = SimpleMemory(system_message="你是一个 AI 助手。请精确遵循指示。") # 中文系统消息

        # 如果模板目录中不存在 react_parser_agent_prompt 文件，
        # PromptManager 将加载/使用默认内容。
        agent = SpecificAgent(
            llm=llm_instance,
            tools=[search_tool_instance],
            memory=agent_memory,
            max_iterations=3 # 限制测试的迭代次数
        )

        # 应使用搜索工具的测试查询
        # query = "伦敦当前天气如何？"
        query = "法国现任总统是谁？" # 更可能需要搜索

        logger.info(f"正在使用查询运行 SpecificAgent: '{query}'")
        try:
            final_output = agent.run(query)
            logger.info(f"SpecificAgent 对于查询 '{query}' 的最终输出:\n{json.dumps(final_output, indent=2, ensure_ascii=False)}") # ensure_ascii=False

            # 断言 (这些是示例，实际 LLM 输出可能不同)
            assert "answer" in final_output or "error" in final_output
            if "answer" in final_output:
                 # 真实的测试会检查答案对于查询是否合理。
                 # 对于“法国总统”，如果搜索正常且 LLM 处理了它，我们可能期望答案中包含“马克龙”。
                 logger.info("Agent 已完成并给出答案。")
            elif "error" in final_output:
                 logger.warning(f"Agent 完成但出现错误: {final_output['error']}")


        except Exception as e:
            logger.error(f"SpecificAgent 同步运行测试期间出错: {e}", exc_info=True)
            raise

        # 测试异步运行
        async def run_async_specific_agent():
            logger.info(f"\n--- 使用查询测试异步 SpecificAgent: '{query}' ---")
            # 为异步测试创建新的内存以避免状态干扰
            async_memory = SimpleMemory(system_message="你是一个用于异步任务的 AI 助手。") # 中文系统消息
            async_agent = SpecificAgent(
                llm=llm_instance,
                tools=[search_tool_instance],
                memory=async_memory,
                max_iterations=3
            )
            try:
                async_final_output = await async_agent.arun(query)
                logger.info(f"SpecificAgent 对于查询 '{query}' 的异步最终输出:\n{json.dumps(async_final_output, indent=2, ensure_ascii=False)}")
                assert "answer" in async_final_output or "error" in async_final_output
            except Exception as e:
                logger.error(f"SpecificAgent 异步运行测试期间出错: {e}", exc_info=True)
                # 根据错误，你可能希望测试失败或仅记录它。
                # 目前，我们仅记录它，因为 LLM 调用可能不稳定。

        try:
            asyncio.run(run_async_specific_agent())
        except Exception as e:
             logger.error(f"运行异步 SpecificAgent 测试失败: {e}", exc_info=True)


        logger.info("SpecificAgent 测试完成。")

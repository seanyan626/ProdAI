# src/agents/base_agent.py
# Agent 的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union

from src.llms.base_llm import BaseLLM
from src.memory.base_memory import BaseMemory
from src.memory.simple_memory import SimpleMemory # 默认内存实现
from src.tools.base_tool import BaseTool
from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class AgentAction:
    """表示 Agent 决定采取的行动。"""
    def __init__(self, tool_name: str, tool_input: Dict[str, Any], log: Optional[str] = None):
        self.tool_name = tool_name # 工具名称
        self.tool_input = tool_input # 工具输入
        self.log = log # Agent 对此行动的思考过程或理由

    def __repr__(self):
        return f"AgentAction(tool='{self.tool_name}', input={self.tool_input}, log='{self.log}')"

class AgentFinish:
    """表示 Agent 完成任务时的最终输出。"""
    def __init__(self, output: Dict[str, Any], log: Optional[str] = None):
        self.output = output # 最终答案或结果
        self.log = log # 最终思考或总结

    def __repr__(self):
        return f"AgentFinish(output={self.output}, log='{self.log}')"


class BaseAgent(ABC):
    """
    Agent 的抽象基类。
    Agent 使用 LLM 来决定行动，可以使用工具，并维护记忆。
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_iterations: int = 10, # 最大迭代次数
        agent_prompt_name: Optional[str] = None, # 主要 Agent 提示模板的名称
        **kwargs: Any # 用于其他特定于 Agent 的配置
    ):
        self.llm = llm
        self.tools = tools or []
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.memory = memory or SimpleMemory() # 默认为简单的内存记忆
        self.prompt_manager = prompt_manager or PromptManager() # 默认提示管理器
        self.max_iterations = max_iterations
        self.agent_prompt_name = agent_prompt_name # 例如 "react_agent_prompt"
        self.config = kwargs

        if self.agent_prompt_name and not self.prompt_manager.get_template(self.agent_prompt_name):
            logger.warning(f"在 PromptManager 中未找到 Agent 提示模板 '{self.agent_prompt_name}'。"
                           "缺少主要提示，Agent 可能无法正常工作。")

        logger.info(f"Agent '{self.__class__.__name__}' 已初始化。LLM: {llm.model_name}, "
                    f"工具: {[tool.name for tool in self.tools]}, 最大迭代次数: {max_iterations}")

    @abstractmethod
    def _plan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]], # (AgentAction, observation) 字典列表
    ) -> Union[AgentAction, AgentFinish]:
        """
        Agent 决定下一步行动或是否应完成的核心逻辑。
        这涉及使用当前输入、历史记录和可用工具来提示 LLM。

        参数:
            inputs (Dict[str, Any]): Agent 的初始输入 (例如用户查询)。
            intermediate_steps (List[Dict[str, Any]]):
                字典列表，其中每个字典表示过去的一个行动及其结果。
                示例: [{"action": AgentAction(...), "observation": "工具输出字符串"}]

        返回:
            Union[AgentAction, AgentFinish]: 要采取的下一个行动或最终结果。
        """
        pass

    async def _aplan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        规划逻辑的异步版本。
        """
        logger.warning(f"Agent '{self.__class__.__name__}' 没有特定的异步规划逻辑。将回退到同步 _plan。")
        # 这是一个占位符。真正的异步 Agent 需要用异步 LLM 调用来实现此方法。
        # return self._plan(inputs, intermediate_steps)
        raise NotImplementedError(f"{self.__class__.__name__}._aplan() 未实现。")


    def _construct_scratchpad(self, intermediate_steps: List[Dict[str, Any]]) -> str:
        """
        构建 Agent 过去行动和观察的字符串表示形式。
        这通常用作 LLM 提示的一部分。
        示例格式 (ReAct 风格):
        Thought: 我需要使用工具 X。
        Action: tool_X
        Action Input: {"key": "value"}
        Observation: 工具 X 的结果
        Thought: ...
        """
        scratchpad = ""
        for step in intermediate_steps:
            action = step.get("action")
            observation = step.get("observation")
            if action and isinstance(action, AgentAction):
                if action.log: # 如果 Agent 记录了导致行动的思考
                    scratchpad += f"Thought: {action.log}\n" # 思考
                scratchpad += f"Action: {action.tool_name}\n" # 行动
                scratchpad += f"Action Input: {action.tool_input}\n" # 行动输入 (或 json.dumps(action.tool_input))
            if observation is not None:
                scratchpad += f"Observation: {str(observation)}\n" # 观察
        return scratchpad.strip()

    def _get_tool_info_string(self) -> str:
        """
        生成描述可用工具的字符串，用于提示。
        格式:
        工具名称: 工具描述。参数模式: {参数的JSON模式}
        """
        if not self.tools:
            return "没有可用的工具。"

        tool_descs = []
        for tool in self.tools:
            schema_info = tool.get_schema_json()
            schema_str = "没有特定的输入参数。"
            if schema_info and schema_info.get("properties"):
                 props = schema_info["properties"]
                 required = schema_info.get("required", [])
                 arg_descs = []
                 for name, details in props.items():
                     desc = details.get("description", "")
                     typ = details.get("type", "any")
                     is_req = " (必需)" if name in required else ""
                     arg_descs.append(f"  - {name} ({typ}): {desc}{is_req}")
                 if arg_descs:
                    schema_str = "参数:\n" + "\n".join(arg_descs)

            tool_descs.append(f"{tool.name}: {tool.description}\n{schema_str}")
        return "\n\n".join(tool_descs)


    def run(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        运行 Agent 直到它完成或达到最大迭代次数。

        参数:
            initial_input (Union[str, Dict[str, Any]]): Agent 的初始输入或查询。
                                                       如果是字符串，通常包装为 {"input": initial_input}。
            **kwargs: 其他运行时参数。

        返回:
            Dict[str, Any]: Agent 的最终输出 (来自 AgentFinish.output)。
        """
        if isinstance(initial_input, str):
            inputs = {"input": initial_input}
        else:
            inputs = initial_input

        intermediate_steps: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Agent 迭代 {iterations}/{self.max_iterations}")

            # Agent 决定下一步 (行动或完成)
            try:
                agent_decision = self._plan(inputs, intermediate_steps)
            except Exception as e:
                logger.error(f"Agent 规划期间出错: {e}", exc_info=True)
                return {"error": f"Agent 规划失败: {e}", "log": self._construct_scratchpad(intermediate_steps)}


            if isinstance(agent_decision, AgentFinish):
                logger.info(f"Agent 完成。输出: {agent_decision.output}。日志: {agent_decision.log}")
                # 如果需要，可以选择将最终思考添加到内存
                self.memory.add_message({"role": "assistant", "content": f"最终答案: {agent_decision.output.get('answer', str(agent_decision.output))}\n推理过程: {agent_decision.log}"})
                return agent_decision.output

            if isinstance(agent_decision, AgentAction):
                tool_name = agent_decision.tool_name
                tool_input = agent_decision.tool_input
                logger.info(f"Agent 行动: 工具: {tool_name}, 输入: {tool_input}, 日志: {agent_decision.log}")

                # 将 Agent 的思考和行动添加到内存/暂存区
                # 格式取决于 _plan 如何期望 intermediate_steps
                # 现在，我们存储行动，接下来将添加观察结果。
                # Agent 自身“思考”或“行动”消息的内存更新
                action_log_msg = f"思考: {agent_decision.log}\n" if agent_decision.log else ""
                action_log_msg += f"行动: 使用工具 {tool_name}，输入 {tool_input}"
                self.memory.add_message({"role": "assistant", "content": action_log_msg}) # 或自定义角色，如 "agent_thought"

                if tool_name in self.tool_map:
                    tool_to_use = self.tool_map[tool_name]
                    try:
                        observation = tool_to_use.run(tool_input)
                        logger.info(f"工具 '{tool_name}' 的观察结果: {str(observation)[:200]}...")
                    except Exception as e:
                        logger.error(f"运行工具 '{tool_name}' 出错: {e}", exc_info=True)
                        observation = f"错误: 运行工具 {tool_name} 失败。详情: {e}"
                else:
                    logger.warning(f"Agent 尝试使用未知工具: {tool_name}")
                    observation = f"错误: 未找到工具 '{tool_name}'。"

                # 将观察结果添加到内存
                self.memory.add_message({"role": "system", "content": f"工具观察 ({tool_name}): {str(observation)}" }) # 或角色 "tool_observation"

                intermediate_steps.append({"action": agent_decision, "observation": observation})
            else:
                logger.error(f"Agent 计划返回无效类型: {type(agent_decision)}。应为 AgentAction 或 AgentFinish。")
                return {"error": "Agent 规划返回意外类型。", "log": self._construct_scratchpad(intermediate_steps)}

        logger.warning(f"Agent 达到最大迭代次数 ({self.max_iterations}) 仍未完成。")
        return {"error": "Agent 因达到最大迭代次数而停止。", "log": self._construct_scratchpad(intermediate_steps)}


    async def arun(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        异步运行 Agent。
        """
        if isinstance(initial_input, str):
            inputs = {"input": initial_input}
        else:
            inputs = initial_input

        intermediate_steps: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Agent 异步迭代 {iterations}/{self.max_iterations}")

            try:
                agent_decision = await self._aplan(inputs, intermediate_steps)
            except NotImplementedError:
                logger.error(f"Agent '{self.__class__.__name__}' 不支持通过 arun 进行异步执行，因为 _aplan 未实现。")
                return {"error": f"Agent {self.__class__.__name__} 不支持异步执行。"}
            except Exception as e:
                logger.error(f"Agent 异步规划期间出错: {e}", exc_info=True)
                return {"error": f"Agent 异步规划失败: {e}", "log": self._construct_scratchpad(intermediate_steps)}

            if isinstance(agent_decision, AgentFinish):
                logger.info(f"Agent (异步) 完成。输出: {agent_decision.output}。日志: {agent_decision.log}")
                await self.memory.add_message({"role": "assistant", "content": f"最终答案: {agent_decision.output.get('answer', str(agent_decision.output))}\n推理过程: {agent_decision.log}"}) # 假设内存具有异步 add 方法
                return agent_decision.output

            if isinstance(agent_decision, AgentAction):
                tool_name = agent_decision.tool_name
                tool_input = agent_decision.tool_input
                logger.info(f"Agent (异步) 行动: 工具: {tool_name}, 输入: {tool_input}, 日志: {agent_decision.log}")

                action_log_msg = f"思考: {agent_decision.log}\n" if agent_decision.log else ""
                action_log_msg += f"行动: 使用工具 {tool_name}，输入 {tool_input}"
                await self.memory.add_message({"role": "assistant", "content": action_log_msg})


                if tool_name in self.tool_map:
                    tool_to_use = self.tool_map[tool_name]
                    try:
                        observation = await tool_to_use.arun(tool_input) # 使用异步工具运行
                        logger.info(f"工具 '{tool_name}' (异步) 的观察结果: {str(observation)[:200]}...")
                    except Exception as e:
                        logger.error(f"运行工具 '{tool_name}' (异步) 出错: {e}", exc_info=True)
                        observation = f"错误: 运行工具 {tool_name} (异步) 失败。详情: {e}"
                else:
                    logger.warning(f"Agent (异步) 尝试使用未知工具: {tool_name}")
                    observation = f"错误: 未找到工具 '{tool_name}'。"

                await self.memory.add_message({"role": "system", "content": f"工具观察 ({tool_name}): {str(observation)}"})
                intermediate_steps.append({"action": agent_decision, "observation": observation})
            else:
                logger.error(f"Agent (异步) 计划返回无效类型: {type(agent_decision)}。应为 AgentAction 或 AgentFinish。")
                return {"error": "Agent (异步) 规划返回意外类型。", "log": self._construct_scratchpad(intermediate_steps)}

        logger.warning(f"Agent (异步) 达到最大迭代次数 ({self.max_iterations}) 仍未完成。")
        return {"error": "Agent (异步) 因达到最大迭代次数而停止。", "log": self._construct_scratchpad(intermediate_steps)}


if __name__ == '__main__':
    # 这是一个概念性测试。需要一个具体的 Agent 实现。
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM # 用于具体的 LLM
    from src.tools.search_tool import SearchTool # 示例工具
    import asyncio

    load_config()
    setup_logging()

    logger.info("BaseAgent 的概念性测试...")

    # 用于测试目的的一个非常基础的具体 Agent (例如 ReAct 风格的 Agent)
    class MyTestAgent(BaseAgent):
        # 这是一个简化的 Agent，它总是尝试使用一次搜索工具，然后完成。
        # 真正的 Agent 会有更复杂的 LLM 提示和解析逻辑。
        def __init__(self, llm: BaseLLM, tools: List[BaseTool], **kwargs):
            # 对于此测试，我们将创建一个虚拟的 Agent 提示模板内容
            # 真正的 Agent 会通过 PromptManager 从文件加载
            self.agent_prompt_content = """
工具:
$tool_descriptions

你可以使用以上工具来回答用户的问题。
你的思考过程应该是：
Thought: [你的推理过程]
Action: [要使用的工具名称，例如 web_search]
Action Input: [工具的输入，JSON 兼容的字典字符串，例如 {"query": "某个查询"}]
Observation: [工具返回的结果]
... (重复 Thought/Action/Action Input/Observation N 次)
Thought: 我现在有足够的信息来回答问题了。
Final Answer: [你对原始问题的最终答案]

当前对话:
$chat_history

用户问题: $input
暂存区 (你的思考、行动和观察):
$scratchpad
Thought:
"""
            # 创建一个虚拟的提示管理器并添加此模板
            pm = kwargs.get("prompt_manager", PromptManager())
            agent_prompt_name = "my_test_agent_prompt_我的测试Agent提示" # 模板名称
            if not pm.get_template(agent_prompt_name):
                pm.loaded_templates[agent_prompt_name] = pm.PromptTemplate(self.agent_prompt_content)

            super().__init__(llm, tools, prompt_manager=pm, agent_prompt_name=agent_prompt_name, **kwargs)


        def _plan(self, inputs: Dict[str, Any], intermediate_steps: List[Dict[str, Any]]) -> Union[AgentAction, AgentFinish]:

            # 测试的简化逻辑:
            # 如果没有中间步骤，尝试搜索。
            # 如果有观察结果，尝试完成。

            current_input = inputs.get("input", "")
            scratchpad_str = self._construct_scratchpad(intermediate_steps)
            tool_info_str = self._get_tool_info_string()
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory.get_history()])


            # 为 LLM 构建提示 (非常简化)
            # 真正的 Agent 会使用 PromptManager 和合适的模板
            prompt_vars = {
                "tool_descriptions": tool_info_str,
                "chat_history": chat_history_str,
                "input": current_input,
                "scratchpad": scratchpad_str
            }
            full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)

            if not full_prompt:
                 return AgentFinish({"answer": "错误: 无法创建 Agent 提示。"}, log="提示格式化失败。")


            logger.debug(f"--- Agent 发送给 LLM 的提示 ---\n{full_prompt}\n--------------------------")

            # 根据当前状态模拟 LLM 响应 (在此基础测试的 _plan 中没有实际的 LLM 调用)
            # 真正的 Agent 会调用: response_text = self.llm.generate(full_prompt) 或 self.llm.chat(...)
            # 然后解析 `response_text` 以提取 Action/Input 或 Final Answer。

            if not intermediate_steps: # 第一步，决定搜索
                # LLM 在这里会生成: "Thought: 我需要搜索 XYZ。\nAction: web_search\nAction Input: {\"query\": \"XYZ\"}"
                # 我们为此测试硬编码:
                action_tool_name = "web_search" # 假设 web_search 工具可用
                if action_tool_name not in self.tool_map:
                    return AgentFinish({"answer": f"无法继续，找不到必需的工具 '{action_tool_name}'。"}, log=f"工具 {action_tool_name} 缺失。")

                simulated_llm_output_for_action = (
                    f"Thought: 我应该搜索关于 '{current_input}' 的信息。\n"
                    f"Action: {action_tool_name}\n"
                    f"Action Input: {{\"query\": \"{current_input}\"}}" # 简化的类 JSON 字符串
                )
                # --- 解析逻辑 (在真正的 Agent 中会很复杂) ---
                # 对于测试，直接从硬编码的模拟输出创建行动
                # 在真正的 Agent 中: 解析 simulated_llm_output_for_action 以提取工具名称和输入
                parsed_tool_name = action_tool_name
                # LLM 可能输出这样的字符串:
                parsed_tool_input_str = f'{{"query": "{current_input}"}}'
                import json
                try:
                    parsed_tool_input_dict = json.loads(parsed_tool_input_str)
                except json.JSONDecodeError:
                    return AgentFinish({"answer": "错误: LLM 为工具输入生成了无效的 JSON。"}, log="LLM 输出解析错误。")

                return AgentAction(
                    tool_name=parsed_tool_name,
                    tool_input=parsed_tool_input_dict,
                    log=f"我应该搜索关于 '{current_input}' 的信息。"
                )
            else: # 第二步 (获得观察结果后)，决定完成
                last_observation = intermediate_steps[-1].get("observation", "未找到观察结果。")
                # LLM 在这里会生成: "Thought: 我有搜索结果了。现在可以回答了。\nFinal Answer: 答案基于..."
                # 我们硬编码:
                simulated_llm_output_for_finish = (
                    f"Thought: 我已收到观察结果: '{str(last_observation)[:50]}...'。现在可以提供答案了。\n"
                    f"Final Answer: 根据搜索，关于 '{current_input}' 的信息是: {str(last_observation)[:100]}..."
                )
                # --- 解析逻辑 ---
                # 对于测试，直接从硬编码的模拟输出创建完成动作
                final_answer_content = f"根据搜索，关于 '{current_input}' 的信息是: {str(last_observation)[:100]}..."
                return AgentFinish(
                    output={"answer": final_answer_content, "source_observation": last_observation},
                    log="我已收到观察结果，现在可以提供答案了。"
                )

        async def _aplan(self, inputs: Dict[str, Any], intermediate_steps: List[Dict[str, Any]]) -> Union[AgentAction, AgentFinish]:
            # 这是一个基础的异步计划，对于更健壮的测试，它应该调用异步 LLM
            # 目前，为简化此基础 Agent 测试，它的行为将与同步版本相同
            logger.info("MyTestAgent._aplan 被调用，此测试使用同步 _plan 逻辑。")
            return self._plan(inputs, intermediate_steps)


    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY 未设置或为占位符。正在跳过使用 MyTestAgent 的 BaseAgent 集成测试。")
    else:
        logger.info("\n--- 使用 MyTestAgent 测试 BaseAgent ---")
        # 设置
        test_llm = OpenAILLM(model_name="gpt-3.5-turbo") # 实际的 LLM (尽管上面的 _plan 模拟了其输出)
        search_tool_instance = SearchTool() # 模拟的搜索工具
        test_agent = MyTestAgent(llm=test_llm, tools=[search_tool_instance], max_iterations=3)

        # 运行 Agent
        agent_query = "关于 AI 的最新消息"
        logger.info(f"正在使用查询运行 MyTestAgent: '{agent_query}'")
        final_result = test_agent.run(agent_query)

        logger.info(f"MyTestAgent 最终结果: {final_result}")
        assert "answer" in final_result
        assert agent_query in final_result.get("answer", "")
        assert "模拟搜索结果" in str(final_result.get("source_observation", "")) # 检查搜索结果是否在答案中 (取决于SearchTool的模拟输出)

        # 测试异步运行 (将使用简化的 _aplan)
        async def run_async_agent_test():
            logger.info(f"\n--- 使用 MyTestAgent 测试异步 BaseAgent ---")
            async_agent_query = "异步 AI 发展"
            async_final_result = await test_agent.arun(async_agent_query) # _aplan 在这里会调用同步 _plan

            logger.info(f"MyTestAgent 异步最终结果: {async_final_result}")
            assert "answer" in async_final_result
            assert async_agent_query in async_final_result.get("answer", "")
            assert "模拟搜索结果" in str(async_final_result.get("source_observation", ""))

        asyncio.run(run_async_agent_test())

    logger.info("BaseAgent 概念性测试完成。")

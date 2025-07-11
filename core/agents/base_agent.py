# src/agents/base_agent.py
# Agent 的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union

from core.models.language.base_language_model import BaseLanguageModel
from core.memory.base_memory import BaseMemory
from core.memory.simple_memory import SimpleMemory # 默认内存实现
from core.tools.base_tool import BaseTool
from core.prompts.prompt_manager import PromptManager

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
    子类应实现具体的规划（_plan）逻辑。
    """

    def __init__(
        self,
        llm: BaseLanguageModel, # <--- 已更改
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_iterations: int = 10,
        agent_prompt_name: Optional[str] = None,
        **kwargs: Any
    ):
        self.llm = llm
        self.tools = tools or []
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.memory = memory or SimpleMemory()
        self.prompt_manager = prompt_manager or PromptManager()
        self.max_iterations = max_iterations
        self.agent_prompt_name = agent_prompt_name
        self.config = kwargs

        # if self.agent_prompt_name and not self.prompt_manager.get_template(self.agent_prompt_name):
        #     logger.warning(f"在 PromptManager 中未找到 Agent 提示模板 '{self.agent_prompt_name}'。"
        #                    "缺少主要提示，Agent 可能无法正常工作。")

        logger.info(f"Agent '{self.__class__.__name__}' 已初始化。语言模型: {llm.model_name}, " # <--- "LLM" -> "语言模型"
                    f"工具: {[tool.name for tool in self.tools]}, 最大迭代次数: {max_iterations}")

    @abstractmethod
    def _plan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        Agent 决定下一步行动或是否应完成的核心逻辑。
        （实现待补充）
        """
        pass

    async def _aplan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        规划逻辑的异步版本。
        （实现待补充）
        """
        # logger.warning(f"Agent '{self.__class__.__name__}' 没有特定的异步规划逻辑。将回退到同步 _plan。")
        # raise NotImplementedError(f"{self.__class__.__name__}._aplan() 未实现。")
        pass


    def _construct_scratchpad(self, intermediate_steps: List[Dict[str, Any]]) -> str:
        """
        构建 Agent 过去行动和观察的字符串表示形式。
        （实现待补充或根据具体 Agent 调整）
        """
        # scratchpad = ""
        # for step in intermediate_steps:
        #     action = step.get("action")
        #     observation = step.get("observation")
        #     if action and isinstance(action, AgentAction):
        #         if action.log:
        #             scratchpad += f"思考: {action.log}\n"
        #         scratchpad += f"行动: {action.tool_name}\n"
        #         scratchpad += f"行动输入: {action.tool_input}\n"
        #     if observation is not None:
        #         scratchpad += f"观察: {str(observation)}\n"
        # return scratchpad.strip()
        return "模拟的暂存区内容。" # 占位符

    def _get_tool_info_string(self) -> str:
        """
        生成描述可用工具的字符串，用于提示。
        （实现待补充）
        """
        # if not self.tools:
        #     return "没有可用的工具。"
        # tool_descs = []
        # # ... (具体格式化逻辑)
        # return "\n\n".join(tool_descs)
        return "模拟的工具信息字符串。" # 占位符


    def run(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        运行 Agent 直到它完成或达到最大迭代次数。
        （实现待补充）
        """
        logger.info(f"BaseAgent.run 调用，初始输入: {str(initial_input)[:50]}...")
        # 循环、调用 _plan、执行工具、处理 AgentFinish 的逻辑将在此处
        return {"answer": "来自 BaseAgent.run 的模拟最终答案。"} # 占位符

    async def arun(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        异步运行 Agent。
        （实现待补充）
        """
        logger.info(f"BaseAgent.arun 调用，初始输入: {str(initial_input)[:50]}...")
        # 类似的异步循环逻辑
        return {"answer": "来自 BaseAgent.arun 的模拟异步最终答案。"} # 占位符


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
    logger.info("BaseAgent 模块。这是一个抽象基类，通常不直接运行。AgentAction 和 AgentFinish 类已定义。")
    pass

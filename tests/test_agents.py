# tests/test_agents.py
# Agent 相关模块的测试
import pytest
import logging
from unittest.mock import MagicMock, patch

# 确保如果任何底层模块需要配置，则加载配置，并设置日志记录。
# 这可能更适合在 conftest.py 中为所有测试统一处理。
try:
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config() # 加载 .env 变量、API 密钥等。
    setup_logging() # 配置测试输出的日志记录
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。这可能会影响模块行为。")


from src.agents.base_agent import BaseAgent, AgentAction, AgentFinish
# from src.agents.specific_agent import SpecificAgent # 具体 Agent 实现待补充
from src.llms.base_llm import BaseLLM
from src.tools.base_tool import BaseTool, ToolInputSchema
from src.memory.simple_memory import SimpleMemory
from src.prompts.prompt_manager import PromptManager


logger = logging.getLogger(__name__)

# --- 模拟对象和测试固件 ---

@pytest.fixture
def mock_llm():
    """模拟 LLM 的测试固件。"""
    llm = MagicMock(spec=BaseLLM)
    llm.model_name = "mock_llm_model_模拟LLM模型"
    llm.generate.return_value = "模拟的LLM单次生成响应。"
    llm.chat.return_value = {"role": "assistant", "content": "模拟的LLM聊天响应。"}
    async def mock_agenerate(*args, **kwargs): return "模拟的异步LLM生成。"
    async def mock_achat(*args, **kwargs): return {"role": "assistant", "content": "模拟的异步LLM聊天。"}
    llm.agenerate = MagicMock(side_effect=mock_agenerate)
    llm.achat = MagicMock(side_effect=mock_achat)
    return llm

class MockToolInput(ToolInputSchema):
    param: str = "default_参数"

class MockTool(BaseTool):
    name: str = "mock_tool_模拟工具"
    description: str = "用于测试的模拟工具。"
    args_schema = MockToolInput

    def _run(self, param: str = "default_参数") -> str:
        return f"模拟工具已执行，参数: {param}"

    async def _arun(self, param: str = "default_参数") -> str:
        return f"异步模拟工具已执行，参数: {param}"

@pytest.fixture
def mock_tool_list():
    """包含一个模拟工具的列表的测试固件。"""
    return [MockTool()]

@pytest.fixture
def simple_memory():
    """SimpleMemory 实例的测试固件。"""
    return SimpleMemory()

@pytest.fixture
def prompt_manager_with_agent_prompt(tmp_path):
    """带有虚拟 Agent 提示的 PromptManager 的测试固件。"""
    templates_dir = tmp_path / "agent_templates_agent模板"
    templates_dir.mkdir()
    agent_prompt_file = templates_dir / "test_agent_prompt.txt"
    agent_prompt_file.write_text("用户: $input\n工具: $tool_descriptions\n暂存区: $scratchpad\n思考:")

    pm = PromptManager(templates_dir=str(templates_dir))
    return pm


# --- 测试 BaseAgent (概念性 - 需要一个具体类或更多模拟) ---
# 注意：由于 SpecificAgent 的实现已被移除，以下测试可能需要调整或针对新的具体 Agent 实现。
# 目前，这些测试将被跳过。

@pytest.mark.skip(reason="BaseAgent 是抽象类，SpecificAgent 实现已移除，测试待针对具体实现更新。")
def test_base_agent_initialization(mock_llm, mock_tool_list, simple_memory):
    logger.info("测试 BaseAgent 初始化 (通过 SpecificAgent 进行概念性测试)。")
    # from src.agents.specific_agent import SpecificAgent # 需要具体实现
    # agent = SpecificAgent(llm=mock_llm, tools=mock_tool_list, memory=simple_memory)
    # assert agent.llm == mock_llm
    # assert agent.tools == mock_tool_list
    # assert agent.memory == simple_memory
    # assert agent.tool_map["mock_tool_模拟工具"] is not None
    pass


# --- 测试 SpecificAgent (具体 Agent 示例) ---
# 注意：SpecificAgent 的实现已被移除，以下测试将失败或需要完全重写。
# 目前，这些测试将被跳过。

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
@pytest.fixture
def specific_agent_components(mock_llm, mock_tool_list, simple_memory, prompt_manager_with_agent_prompt):
    # from src.agents.specific_agent import SpecificAgent # 需要具体实现
    # default_agent_prompt_name = "react_parser_agent_prompt"
    # agent_prompt_file = prompt_manager_with_agent_prompt.templates_dir / f"{default_agent_prompt_name}.txt"
    # agent_prompt_file.write_text(
    #     "工具: $tool_descriptions\n输入: $input\n暂存区: $scratchpad\n思考:"
    # )
    # reloaded_pm = PromptManager(templates_dir=prompt_manager_with_agent_prompt.templates_dir)
    # return {
    #     "llm": mock_llm,
    #     "tools": mock_tool_list,
    #     "memory": simple_memory,
    #     "prompt_manager": reloaded_pm
    # }
    return {} # 返回空字典，因为依赖的类可能不存在

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
def test_specific_agent_initialization(specific_agent_components):
    logger.info("测试 SpecificAgent 初始化。")
    # from src.agents.specific_agent import SpecificAgent # 需要具体实现
    # if not specific_agent_components: pytest.skip("组件未正确初始化，跳过测试")
    # agent = SpecificAgent(**specific_agent_components)
    # assert agent is not None
    pass

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
def test_specific_agent_plan_parses_final_answer(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 方法的 Final Answer 解析。")
    # ... 原测试逻辑 ...
    pass

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
def test_specific_agent_plan_parses_action(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 方法的 AgentAction 解析。")
    # ... 原测试逻辑 ...
    pass

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
def test_specific_agent_plan_handles_malformed_json_input(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 处理 Action Input 中格式错误的 JSON。")
    # ... 原测试逻辑 ...
    pass

@pytest.mark.skip(reason="SpecificAgent 实现已移除，测试待更新。")
def test_specific_agent_run_one_step_and_finish(specific_agent_components, mock_tool_list):
    logger.info("测试 SpecificAgent run 方法进行一次工具调用然后完成。")
    # ... 原测试逻辑 ...
    pass

# TODO: 针对骨架结构，可以添加一些基础的导入测试或接口存在性测试。
# 例如，测试 BaseAgent, AgentAction, AgentFinish 是否可以被导入。
def test_base_classes_importable():
    logger.info("测试基础 Agent 类是否可导入。")
    try:
        from src.agents.base_agent import BaseAgent, AgentAction, AgentFinish
        assert BaseAgent is not None
        assert AgentAction is not None
        assert AgentFinish is not None
        logger.info("基础 Agent 类成功导入。")
    except ImportError as e:
        pytest.fail(f"无法导入基础 Agent 类: {e}")

logger.info("Agent 测试文件已调整为适应骨架代码。多数原测试已跳过。")

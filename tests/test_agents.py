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
from src.agents.specific_agent import SpecificAgent # 假设 SpecificAgent 是一个具体实现
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
    # 模拟 generate 和 chat 方法以返回可预测的响应
    llm.generate.return_value = "模拟的LLM单次生成响应。"
    llm.chat.return_value = {"role": "assistant", "content": "模拟的LLM聊天响应。"}
    # 异步版本
    async def mock_agenerate(*args, **kwargs): return "模拟的异步LLM生成。"
    async def mock_achat(*args, **kwargs): return {"role": "assistant", "content": "模拟的异步LLM聊天。"}
    llm.agenerate = MagicMock(side_effect=mock_agenerate)
    llm.achat = MagicMock(side_effect=mock_achat)
    return llm

class MockToolInput(ToolInputSchema):
    param: str = "default_参数" # 模拟工具输入参数

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
    agent_prompt_file = templates_dir / "test_agent_prompt.txt" # 测试agent提示文件
    agent_prompt_file.write_text("用户: $input\n工具: $tool_descriptions\n暂存区: $scratchpad\n思考:")

    pm = PromptManager(templates_dir=str(templates_dir))
    return pm


# --- 测试 BaseAgent (概念性 - 需要具体类或更多模拟) ---

def test_base_agent_initialization(mock_llm, mock_tool_list, simple_memory):
    logger.info("测试 BaseAgent 初始化 (通过 SpecificAgent 进行概念性测试)。")
    # BaseAgent 是抽象的，因此我们可能通过具体子类测试其初始化，
    # 或创建一个用于测试的虚拟具体类。这里使用 SpecificAgent。
    try:
        agent = SpecificAgent(llm=mock_llm, tools=mock_tool_list, memory=simple_memory)
        assert agent.llm == mock_llm
        assert agent.tools == mock_tool_list
        assert agent.memory == simple_memory
        assert agent.tool_map["mock_tool_模拟工具"] is not None # 检查模拟工具是否在映射中
    except Exception as e:
        pytest.fail(f"Agent 初始化失败: {e}")


# --- 测试 SpecificAgent (具体 Agent 示例) ---

@pytest.fixture
def specific_agent_components(mock_llm, mock_tool_list, simple_memory, prompt_manager_with_agent_prompt):
    # 使用定义了 test_agent_prompt 的提示管理器，
    # 并告诉 SpecificAgent 使用该提示名称。
    # SpecificAgent 的默认提示名称是 'react_parser_agent_prompt'。
    # 对于此测试，我们将使用自定义名称以确保从我们的固件加载。
    # 或者，我们可以模拟 SpecificAgent 的默认提示名称以匹配我们的测试文件。

    # 最简单的方法：确保提示管理器具有 SpecificAgent 的默认提示
    default_agent_prompt_name = "react_parser_agent_prompt" # SpecificAgent 的默认值
    agent_prompt_file = prompt_manager_with_agent_prompt.templates_dir / f"{default_agent_prompt_name}.txt"
    agent_prompt_file.write_text(
        "工具: $tool_descriptions\n输入: $input\n暂存区: $scratchpad\n思考:" # 简化的提示内容
    )
    # 重新初始化提示管理器以拾取新文件 (或确保在 PM 动态加载时已加载)
    # 有点取巧；如果 PM 有 reload 或 add_template 方法会更好。
    # 为简单起见，假设 PromptManager 在初始化时加载所有内容。
    reloaded_pm = PromptManager(templates_dir=prompt_manager_with_agent_prompt.templates_dir)

    return {
        "llm": mock_llm,
        "tools": mock_tool_list,
        "memory": simple_memory,
        "prompt_manager": reloaded_pm # 使用带有提示的那个
    }

def test_specific_agent_initialization(specific_agent_components):
    logger.info("测试 SpecificAgent 初始化。")
    try:
        agent = SpecificAgent(**specific_agent_components)
        assert agent is not None
        assert agent.agent_prompt_name == "react_parser_agent_prompt" # 默认值
        assert agent.prompt_manager.get_template(agent.agent_prompt_name) is not None
    except Exception as e:
        pytest.fail(f"SpecificAgent 初始化失败: {e}")


def test_specific_agent_plan_parses_final_answer(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 方法的 Final Answer 解析。")
    agent = SpecificAgent(**specific_agent_components)

    # 模拟 LLM 返回应解析为 AgentFinish 的响应
    mock_llm_output = "Thought: 我有足够的信息了。\nFinal Answer: 结果是 42。" # 中文思考和答案
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output} # 如果使用 chat
    agent.llm.generate.return_value = mock_llm_output # 如果使用 generate

    inputs = {"input": "生命的意义是什么？"}
    intermediate_steps = []

    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentFinish)
    assert decision.output == {"answer": "结果是 42。"}
    assert "我有足够的信息了" in decision.log


def test_specific_agent_plan_parses_action(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 方法的 AgentAction 解析。")
    agent = SpecificAgent(**specific_agent_components)

    # 模拟 LLM 返回一个行动响应
    mock_llm_output = 'Thought: 我需要使用模拟工具。\nAction: mock_tool_模拟工具\nAction Input: {"param": "测试值"}' # 中文思考和参数值
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output}
    agent.llm.generate.return_value = mock_llm_output

    inputs = {"input": "使用模拟工具。"}
    intermediate_steps = []

    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentAction)
    assert decision.tool_name == "mock_tool_模拟工具"
    assert decision.tool_input == {"param": "测试值"}
    assert "我需要使用模拟工具" in decision.log


def test_specific_agent_plan_handles_malformed_json_input(specific_agent_components):
    logger.info("测试 SpecificAgent _plan 处理 Action Input 中格式错误的 JSON。")
    agent = SpecificAgent(**specific_agent_components)

    # LLM 返回一个带有格式错误 JSON 的行动
    mock_llm_output = 'Thought: 正在尝试一个行动。\nAction: mock_tool_模拟工具\nAction Input: {"param": "测试值",,}' # 多余的逗号
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output}
    agent.llm.generate.return_value = mock_llm_output

    inputs = {"input": "测试格式错误的 JSON。"}
    intermediate_steps = []
    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentFinish) # 应以错误结束
    assert "error" in decision.output
    assert "invalid json" in decision.output["error"].lower() # 检查英文 "invalid json" 因为那是Python json库的错误信息


def test_specific_agent_run_one_step_and_finish(specific_agent_components, mock_tool_list):
    logger.info("测试 SpecificAgent run 方法进行一次工具调用然后完成。")
    agent = SpecificAgent(**specific_agent_components) # 使用模拟的 LLM 和工具

    # --- 设置 LLM 模拟以引导 Agent ---
    # 1. 第一次 LLM 调用 (规划): Agent 决定使用 mock_tool
    llm_response_action = 'Thought: 我应该使用 mock_tool_模拟工具。\nAction: mock_tool_模拟工具\nAction Input: {"param": "运行测试"}'

    # 2. 第二次 LLM 调用 (工具观察后): Agent 决定完成
    # mock_tool 的观察结果将是 "模拟工具已执行，参数: 运行测试"
    # 此观察结果将成为第二次 LLM 调用的暂存区的一部分。
    llm_response_finish = "Thought: 我已获得工具的结果，现在可以回答了。\nFinal Answer: 模拟工具说：模拟工具已执行，参数: 运行测试"

    # 配置模拟 LLM 以按顺序返回这些响应
    agent.llm.chat.side_effect = [
        {"role": "assistant", "content": llm_response_action},
        {"role": "assistant", "content": llm_response_finish}
    ]
    agent.llm.generate.side_effect = [ # 如果 agent 使用 generate
        llm_response_action,
        llm_response_finish
    ]

    # --- 运行 Agent ---
    initial_query = "运行模拟工具并告诉我它说了什么。"
    final_result = agent.run(initial_query)

    # --- 断言 ---
    assert "answer" in final_result, f"Agent 未提供答案。结果: {final_result}"
    assert "模拟工具已执行，参数: 运行测试" in final_result["answer"]

    # 检查 LLM 是否被调用了两次 (一次用于行动，一次用于完成)
    # 这取决于 Agent 是使用 chat 还是 generate。
    if hasattr(agent.llm, 'chat') and agent.llm.chat.called:
        assert agent.llm.chat.call_count == 2
    elif hasattr(agent.llm, 'generate') and agent.llm.generate.called:
        assert agent.llm.generate.call_count == 2

    # 检查模拟工具是否被调用了一次
    # 我们需要一种方法来断言这一点。如果工具是 MagicMock，我们可以检查 call_count。
    # 在这里，MockTool 是一个真实的类。我们可以 patch 其 _run 方法。
    with patch.object(MockTool, '_run', wraps=mock_tool_list[0]._run) as patched_tool_run:
        # 如有必要，使用 patch 后的工具重新运行，或确保先前的运行使用了此实例。
        # 为简单起见，假设先前的运行确实使用了它。
        # 此断言需要 Agent 使用的工具实例。
        # Agent 创建自己的 tool_map。
        # 更好的方法：为此测试将 MagicMock 工具传递给 Agent。

        # 让我们使用 MagicMock 工具重新运行以获得更清晰的断言
        magic_mock_tool = MagicMock(spec=MockTool)
        magic_mock_tool.name = "mock_tool_模拟工具" # 确保名称匹配
        magic_mock_tool.description = "一个模拟工具。"
        magic_mock_tool.args_schema = MockToolInput
        magic_mock_tool.run.return_value = "MagicMock工具已执行，参数: 运行测试"

        # 使用此 MagicMock 工具创建新的 Agent 实例
        agent_with_magic_tool_components = specific_agent_components.copy()
        agent_with_magic_tool_components["tools"] = [magic_mock_tool]
        agent_with_magic_tool = SpecificAgent(**agent_with_magic_tool_components)

        # 为新 Agent 重置 LLM 调用计数
        agent_with_magic_tool.llm.chat.reset_mock()
        agent_with_magic_tool.llm.generate.reset_mock()
        # 更新模拟LLM的响应内容，以匹配新的模拟工具名称和输出
        llm_response_action_magic = f'Thought: 我应该使用 {magic_mock_tool.name}。\nAction: {magic_mock_tool.name}\nAction Input: {{"param": "运行测试"}}'
        llm_response_finish_magic = f"Thought: 好的。\nFinal Answer: MagicMock工具说：MagicMock工具已执行，参数: 运行测试"

        agent_with_magic_tool.llm.chat.side_effect = [
            {"role": "assistant", "content": llm_response_action_magic},
            {"role": "assistant", "content": llm_response_finish_magic}
        ]
        agent_with_magic_tool.llm.generate.side_effect = [
             llm_response_action_magic,
             llm_response_finish_magic
        ]

        final_result_magic_tool = agent_with_magic_tool.run(initial_query)

        magic_mock_tool.run.assert_called_once_with({"param": "运行测试"})
        assert "MagicMock工具已执行" in final_result_magic_tool["answer"]


# TODO: 添加异步 Agent 执行的测试 (_aplan, arun)
# TODO: 添加最大迭代次数限制的测试
# TODO: 添加 _parse_llm_output 中特定解析边缘情况的测试
# TODO: 添加工具未找到场景的测试

logger.info("Agent 测试完成 (基本结构)。")

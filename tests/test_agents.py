# tests/test_agents.py
import pytest
import logging
from unittest.mock import MagicMock, patch

# Ensure config is loaded if any underlying modules need it, and logging is set up.
# This might be better handled in a conftest.py for all tests.
try:
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config() # Load .env variables, API keys etc.
    setup_logging() # Configure logging for test outputs
except ImportError:
    print("Warning: Could not import config/logging for tests. This might affect module behavior.")


from src.agents.base_agent import BaseAgent, AgentAction, AgentFinish
from src.agents.specific_agent import SpecificAgent # Assuming SpecificAgent is a concrete implementation
from src.llms.base_llm import BaseLLM
from src.tools.base_tool import BaseTool, ToolInputSchema
from src.memory.simple_memory import SimpleMemory
from src.prompts.prompt_manager import PromptManager


logger = logging.getLogger(__name__)

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM."""
    llm = MagicMock(spec=BaseLLM)
    llm.model_name = "mock_llm_model"
    # Mock the generate and chat methods to return predictable responses
    llm.generate.return_value = "Mocked LLM single generation response."
    llm.chat.return_value = {"role": "assistant", "content": "Mocked LLM chat response."}
    # async versions
    async def mock_agenerate(*args, **kwargs): return "Mocked async LLM generation."
    async def mock_achat(*args, **kwargs): return {"role": "assistant", "content": "Mocked async LLM chat."}
    llm.agenerate = MagicMock(side_effect=mock_agenerate)
    llm.achat = MagicMock(side_effect=mock_achat)
    return llm

class MockToolInput(ToolInputSchema):
    param: str = "default"

class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing."
    args_schema = MockToolInput

    def _run(self, param: str = "default") -> str:
        return f"MockTool executed with param: {param}"

    async def _arun(self, param: str = "default") -> str:
        return f"Async MockTool executed with param: {param}"

@pytest.fixture
def mock_tool_list():
    """Fixture for a list containing a mock tool."""
    return [MockTool()]

@pytest.fixture
def simple_memory():
    """Fixture for a SimpleMemory instance."""
    return SimpleMemory()

@pytest.fixture
def prompt_manager_with_agent_prompt(tmp_path):
    """Fixture for a PromptManager with a dummy agent prompt."""
    templates_dir = tmp_path / "agent_templates"
    templates_dir.mkdir()
    agent_prompt_file = templates_dir / "test_agent_prompt.txt"
    agent_prompt_file.write_text("User: $input\nTools: $tool_descriptions\nScratchpad: $scratchpad\nThought:")

    pm = PromptManager(templates_dir=str(templates_dir))
    return pm


# --- Test BaseAgent (Conceptual - requires a concrete class or more mocking) ---

def test_base_agent_initialization(mock_llm, mock_tool_list, simple_memory):
    logger.info("Testing BaseAgent initialization (conceptual via SpecificAgent).")
    # BaseAgent is abstract, so we might test its initialization via a concrete subclass
    # or by creating a dummy concrete class for testing. Here, using SpecificAgent.
    try:
        agent = SpecificAgent(llm=mock_llm, tools=mock_tool_list, memory=simple_memory)
        assert agent.llm == mock_llm
        assert agent.tools == mock_tool_list
        assert agent.memory == simple_memory
        assert agent.tool_map["mock_tool"] is not None
    except Exception as e:
        pytest.fail(f"Agent initialization failed: {e}")


# --- Test SpecificAgent (Example concrete agent) ---

@pytest.fixture
def specific_agent_components(mock_llm, mock_tool_list, simple_memory, prompt_manager_with_agent_prompt):
    # Use the prompt manager that has the test_agent_prompt defined
    # And tell SpecificAgent to use that prompt name.
    # SpecificAgent's default prompt name is 'react_parser_agent_prompt'.
    # For this test, we'll use a custom one to ensure it's loaded from our fixture.
    # Or, we can mock SpecificAgent's default prompt name to match our test file.

    # Easiest: ensure the prompt manager has the SpecificAgent's default prompt
    default_agent_prompt_name = "react_parser_agent_prompt" # Default for SpecificAgent
    agent_prompt_file = prompt_manager_with_agent_prompt.templates_dir / f"{default_agent_prompt_name}.txt"
    agent_prompt_file.write_text(
        "Tools: $tool_descriptions\nInput: $input\nScratchpad: $scratchpad\nThought:"
    )
    # Re-initialize prompt manager to pick up the new file (or ensure it's loaded if PM loads dynamically)
    # A bit hacky; better if PM had a reload or add_template method.
    # For simplicity, assume PromptManager loads all on init.
    reloaded_pm = PromptManager(templates_dir=prompt_manager_with_agent_prompt.templates_dir)

    return {
        "llm": mock_llm,
        "tools": mock_tool_list,
        "memory": simple_memory,
        "prompt_manager": reloaded_pm # Use the one with the prompt
    }

def test_specific_agent_initialization(specific_agent_components):
    logger.info("Testing SpecificAgent initialization.")
    try:
        agent = SpecificAgent(**specific_agent_components)
        assert agent is not None
        assert agent.agent_prompt_name == "react_parser_agent_prompt" # Default
        assert agent.prompt_manager.get_template(agent.agent_prompt_name) is not None
    except Exception as e:
        pytest.fail(f"SpecificAgent initialization failed: {e}")


def test_specific_agent_plan_parses_final_answer(specific_agent_components):
    logger.info("Testing SpecificAgent _plan method for Final Answer parsing.")
    agent = SpecificAgent(**specific_agent_components)

    # Mock LLM to return a response that should be parsed as AgentFinish
    mock_llm_output = "Thought: I have enough information.\nFinal Answer: The result is 42."
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output} # if it uses chat
    agent.llm.generate.return_value = mock_llm_output # if it uses generate

    inputs = {"input": "What is the meaning of life?"}
    intermediate_steps = []

    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentFinish)
    assert decision.output == {"answer": "The result is 42."}
    assert "I have enough information" in decision.log


def test_specific_agent_plan_parses_action(specific_agent_components):
    logger.info("Testing SpecificAgent _plan method for AgentAction parsing.")
    agent = SpecificAgent(**specific_agent_components)

    # Mock LLM to return a response for an action
    mock_llm_output = 'Thought: I need to use the mock tool.\nAction: mock_tool\nAction Input: {"param": "test_value"}'
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output}
    agent.llm.generate.return_value = mock_llm_output

    inputs = {"input": "Use the mock tool."}
    intermediate_steps = []

    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentAction)
    assert decision.tool_name == "mock_tool"
    assert decision.tool_input == {"param": "test_value"}
    assert "I need to use the mock tool" in decision.log


def test_specific_agent_plan_handles_malformed_json_input(specific_agent_components):
    logger.info("Testing SpecificAgent _plan with malformed JSON in Action Input.")
    agent = SpecificAgent(**specific_agent_components)

    # LLM returns an action with malformed JSON
    mock_llm_output = 'Thought: Trying an action.\nAction: mock_tool\nAction Input: {"param": "test_value",,}' # Extra comma
    agent.llm.chat.return_value = {"role": "assistant", "content": mock_llm_output}
    agent.llm.generate.return_value = mock_llm_output

    inputs = {"input": "Test malformed JSON."}
    intermediate_steps = []
    decision = agent._plan(inputs, intermediate_steps)

    assert isinstance(decision, AgentFinish) # Should finish with an error
    assert "error" in decision.output
    assert "invalid JSON" in decision.output["error"].lower()


def test_specific_agent_run_one_step_and_finish(specific_agent_components, mock_tool_list):
    logger.info("Testing SpecificAgent run method for one tool call then finish.")
    agent = SpecificAgent(**specific_agent_components) # uses mocked LLM and Tool

    # --- Setup LLM mock to guide the agent ---
    # 1. First LLM call (planning): Agent decides to use mock_tool
    llm_response_action = 'Thought: I should use mock_tool.\nAction: mock_tool\nAction Input: {"param": "run_test"}'

    # 2. Second LLM call (after tool observation): Agent decides to finish
    # The observation from mock_tool will be "MockTool executed with param: run_test"
    # This observation will be part of the scratchpad for the second LLM call.
    llm_response_finish = "Thought: I have the tool's result, now I can answer.\nFinal Answer: The mock tool said: MockTool executed with param: run_test"

    # Configure the mock LLM to return these responses sequentially
    agent.llm.chat.side_effect = [
        {"role": "assistant", "content": llm_response_action},
        {"role": "assistant", "content": llm_response_finish}
    ]
    agent.llm.generate.side_effect = [ # if agent uses generate
        llm_response_action,
        llm_response_finish
    ]

    # --- Run the agent ---
    initial_query = "Run the mock tool and tell me what it says."
    final_result = agent.run(initial_query)

    # --- Assertions ---
    assert "answer" in final_result, f"Agent did not provide an answer. Result: {final_result}"
    assert "MockTool executed with param: run_test" in final_result["answer"]

    # Check if the LLM was called twice (once for action, once for finish)
    # This depends on whether the agent uses chat or generate.
    if hasattr(agent.llm, 'chat') and agent.llm.chat.called:
        assert agent.llm.chat.call_count == 2
    elif hasattr(agent.llm, 'generate') and agent.llm.generate.called:
        assert agent.llm.generate.call_count == 2

    # Check if the mock tool was called once
    # We need a way to assert this. If the tool is a MagicMock, we can check call_count.
    # Here, MockTool is a real class. We could patch its _run method.
    with patch.object(MockTool, '_run', wraps=mock_tool_list[0]._run) as patched_tool_run:
        # Re-run with the patched tool if necessary, or ensure the previous run used this instance.
        # For simplicity, assume the previous run did use it.
        # This assertion needs the tool instance used by the agent.
        # The agent creates its own tool_map.
        # A better way: pass a MagicMock tool into the agent for this test.

        # Let's re-run with a MagicMock tool for clearer assertion
        magic_mock_tool = MagicMock(spec=MockTool)
        magic_mock_tool.name = "mock_tool"
        magic_mock_tool.description = "A mock tool."
        magic_mock_tool.args_schema = MockToolInput
        magic_mock_tool.run.return_value = "MagicMockTool executed with param: run_test"

        # Create a new agent instance with this MagicMock tool
        agent_with_magic_tool_components = specific_agent_components.copy()
        agent_with_magic_tool_components["tools"] = [magic_mock_tool]
        agent_with_magic_tool = SpecificAgent(**agent_with_magic_tool_components)

        # Reset LLM call counts for the new agent
        agent_with_magic_tool.llm.chat.reset_mock()
        agent_with_magic_tool.llm.generate.reset_mock()
        agent_with_magic_tool.llm.chat.side_effect = [
            {"role": "assistant", "content": llm_response_action},
            {"role": "assistant", "content": "Thought: OK.\nFinal Answer: The magic mock tool said: MagicMockTool executed with param: run_test"}
        ]
        agent_with_magic_tool.llm.generate.side_effect = [
             llm_response_action,
             "Thought: OK.\nFinal Answer: The magic mock tool said: MagicMockTool executed with param: run_test"
        ]


        final_result_magic_tool = agent_with_magic_tool.run(initial_query)

        magic_mock_tool.run.assert_called_once_with({"param": "run_test"})
        assert "MagicMockTool executed" in final_result_magic_tool["answer"]


# TODO: Add tests for async agent execution (_aplan, arun)
# TODO: Add tests for max_iterations limit
# TODO: Add tests for specific parsing edge cases in _parse_llm_output
# TODO: Add tests for tool not found scenarios

logger.info("Agent tests completed (basic structure).")

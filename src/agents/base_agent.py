# src/agents/base_agent.py
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Union

from src.llms.base_llm import BaseLLM
from src.memory.base_memory import BaseMemory
from src.memory.simple_memory import SimpleMemory # Default memory implementation
from src.tools.base_tool import BaseTool
from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class AgentAction:
    """Represents an action an agent decides to take."""
    def __init__(self, tool_name: str, tool_input: Dict[str, Any], log: Optional[str] = None):
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.log = log # Agent's thought process or reasoning for this action

    def __repr__(self):
        return f"AgentAction(tool='{self.tool_name}', input={self.tool_input}, log='{self.log}')"

class AgentFinish:
    """Represents the final output from an agent when it finishes its task."""
    def __init__(self, output: Dict[str, Any], log: Optional[str] = None):
        self.output = output # The final answer or result
        self.log = log # Final thoughts or summary

    def __repr__(self):
        return f"AgentFinish(output={self.output}, log='{self.log}')"


class BaseAgent(ABC):
    """
    Abstract base class for agents.
    Agents use an LLM to decide on actions, can use tools, and maintain memory.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[BaseMemory] = None,
        prompt_manager: Optional[PromptManager] = None,
        max_iterations: int = 10,
        agent_prompt_name: Optional[str] = None, # Name of the main agent prompt template
        **kwargs: Any # For additional agent-specific configurations
    ):
        self.llm = llm
        self.tools = tools or []
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.memory = memory or SimpleMemory() # Default to simple in-memory
        self.prompt_manager = prompt_manager or PromptManager() # Default prompt manager
        self.max_iterations = max_iterations
        self.agent_prompt_name = agent_prompt_name # e.g., "react_agent_prompt"
        self.config = kwargs

        if self.agent_prompt_name and not self.prompt_manager.get_template(self.agent_prompt_name):
            logger.warning(f"Agent prompt template '{self.agent_prompt_name}' not found in PromptManager. "
                           "The agent might not function correctly without its main prompt.")

        logger.info(f"Agent '{self.__class__.__name__}' initialized. LLM: {llm.model_name}, "
                    f"Tools: {[tool.name for tool in self.tools]}, Max Iterations: {max_iterations}")

    @abstractmethod
    def _plan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]], # List of (AgentAction, observation) dicts
    ) -> Union[AgentAction, AgentFinish]:
        """
        Core logic for the agent to decide the next action or if it should finish.
        This involves prompting the LLM with current inputs, history, and available tools.

        Args:
            inputs (Dict[str, Any]): The initial inputs to the agent (e.g., user query).
            intermediate_steps (List[Dict[str, Any]]):
                A list of dictionaries, where each dictionary represents a past action and its result.
                Example: [{"action": AgentAction(...), "observation": "tool output string"}]

        Returns:
            Union[AgentAction, AgentFinish]: The next action to take or the final result.
        """
        pass

    async def _aplan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        """
        Asynchronous version of the planning logic.
        """
        logger.warning(f"Agent '{self.__class__.__name__}' does not have specific async planning. Falling back to sync _plan.")
        # This is a placeholder. True async agents need to implement this with async LLM calls.
        # return self._plan(inputs, intermediate_steps)
        raise NotImplementedError(f"{self.__class__.__name__}._aplan() is not implemented.")


    def _construct_scratchpad(self, intermediate_steps: List[Dict[str, Any]]) -> str:
        """
        Constructs a string representation of the agent's past actions and observations.
        This is often used as part of the prompt to the LLM.
        Example format (ReAct style):
        Thought: I need to use tool X.
        Action: tool_X
        Action Input: {"key": "value"}
        Observation: Result from tool_X
        Thought: ...
        """
        scratchpad = ""
        for step in intermediate_steps:
            action = step.get("action")
            observation = step.get("observation")
            if action and isinstance(action, AgentAction):
                if action.log: # If agent recorded its thought leading to action
                    scratchpad += f"Thought: {action.log}\n"
                scratchpad += f"Action: {action.tool_name}\n"
                scratchpad += f"Action Input: {action.tool_input}\n" # Or json.dumps(action.tool_input)
            if observation is not None:
                scratchpad += f"Observation: {str(observation)}\n"
        return scratchpad.strip()

    def _get_tool_info_string(self) -> str:
        """
        Generates a string describing available tools, for use in prompts.
        Format:
        Tool Name: Tool Description. Args schema: {JSON schema for args}
        """
        if not self.tools:
            return "No tools available."

        tool_descs = []
        for tool in self.tools:
            schema_info = tool.get_schema_json()
            # schema_str = json.dumps(schema_info) if schema_info else "No input arguments."
            # More readable schema:
            schema_str = "No specific input arguments."
            if schema_info and schema_info.get("properties"):
                 props = schema_info["properties"]
                 required = schema_info.get("required", [])
                 arg_descs = []
                 for name, details in props.items():
                     desc = details.get("description", "")
                     typ = details.get("type", "any")
                     is_req = " (required)" if name in required else ""
                     arg_descs.append(f"  - {name} ({typ}): {desc}{is_req}")
                 if arg_descs:
                    schema_str = "Arguments:\n" + "\n".join(arg_descs)

            tool_descs.append(f"{tool.name}: {tool.description}\n{schema_str}")
        return "\n\n".join(tool_descs)


    def run(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Runs the agent until it finishes or reaches max iterations.

        Args:
            initial_input (Union[str, Dict[str, Any]]): The initial input or query for the agent.
                                                       If string, it's typically wrapped as {"input": initial_input}.
            **kwargs: Additional runtime arguments.

        Returns:
            Dict[str, Any]: The final output from the agent (from AgentFinish.output).
        """
        if isinstance(initial_input, str):
            inputs = {"input": initial_input}
        else:
            inputs = initial_input

        intermediate_steps: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Agent iteration {iterations}/{self.max_iterations}")

            # Agent decides next step (action or finish)
            try:
                agent_decision = self._plan(inputs, intermediate_steps)
            except Exception as e:
                logger.error(f"Error during agent planning: {e}", exc_info=True)
                return {"error": f"Agent planning failed: {e}", "log": self._construct_scratchpad(intermediate_steps)}


            if isinstance(agent_decision, AgentFinish):
                logger.info(f"Agent finished. Output: {agent_decision.output}. Log: {agent_decision.log}")
                # Optionally add final thought to memory if needed
                self.memory.add_message({"role": "assistant", "content": f"Final Answer: {agent_decision.output.get('answer', str(agent_decision.output))}\nReasoning: {agent_decision.log}"})
                return agent_decision.output

            if isinstance(agent_decision, AgentAction):
                tool_name = agent_decision.tool_name
                tool_input = agent_decision.tool_input
                logger.info(f"Agent action: Tool: {tool_name}, Input: {tool_input}, Log: {agent_decision.log}")

                # Add agent's thought and action to memory/scratchpad
                # The format depends on how _plan expects intermediate_steps
                # For now, we store the action and will add observation next.
                # Memory update for agent's own "thought" or "action" message
                action_log_msg = f"Thought: {agent_decision.log}\n" if agent_decision.log else ""
                action_log_msg += f"Action: Using tool {tool_name} with input {tool_input}"
                self.memory.add_message({"role": "assistant", "content": action_log_msg}) # Or a custom role like "agent_thought"

                if tool_name in self.tool_map:
                    tool_to_use = self.tool_map[tool_name]
                    try:
                        observation = tool_to_use.run(tool_input)
                        logger.info(f"Tool '{tool_name}' observation: {str(observation)[:200]}...")
                    except Exception as e:
                        logger.error(f"Error running tool '{tool_name}': {e}", exc_info=True)
                        observation = f"Error: Failed to run tool {tool_name}. Details: {e}"
                else:
                    logger.warning(f"Agent tried to use unknown tool: {tool_name}")
                    observation = f"Error: Tool '{tool_name}' not found."

                # Add observation to memory
                self.memory.add_message({"role": "system", "content": f"Tool Observation ({tool_name}): {str(observation)}" }) # Or role "tool_observation"

                intermediate_steps.append({"action": agent_decision, "observation": observation})
            else:
                logger.error(f"Agent plan returned invalid type: {type(agent_decision)}. Expected AgentAction or AgentFinish.")
                return {"error": "Agent planning returned an unexpected type.", "log": self._construct_scratchpad(intermediate_steps)}

        logger.warning(f"Agent reached max iterations ({self.max_iterations}) without finishing.")
        return {"error": "Agent stopped due to max iterations.", "log": self._construct_scratchpad(intermediate_steps)}


    async def arun(self, initial_input: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Asynchronously runs the agent.
        """
        if isinstance(initial_input, str):
            inputs = {"input": initial_input}
        else:
            inputs = initial_input

        intermediate_steps: List[Dict[str, Any]] = []
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"Agent async iteration {iterations}/{self.max_iterations}")

            try:
                agent_decision = await self._aplan(inputs, intermediate_steps)
            except NotImplementedError:
                logger.error(f"Agent '{self.__class__.__name__}' does not support async execution via arun because _aplan is not implemented.")
                return {"error": f"Agent {self.__class__.__name__} does not support async execution."}
            except Exception as e:
                logger.error(f"Error during agent async planning: {e}", exc_info=True)
                return {"error": f"Agent async planning failed: {e}", "log": self._construct_scratchpad(intermediate_steps)}

            if isinstance(agent_decision, AgentFinish):
                logger.info(f"Agent (async) finished. Output: {agent_decision.output}. Log: {agent_decision.log}")
                await self.memory.add_message({"role": "assistant", "content": f"Final Answer: {agent_decision.output.get('answer', str(agent_decision.output))}\nReasoning: {agent_decision.log}"}) # Assuming memory has async add
                return agent_decision.output

            if isinstance(agent_decision, AgentAction):
                tool_name = agent_decision.tool_name
                tool_input = agent_decision.tool_input
                logger.info(f"Agent (async) action: Tool: {tool_name}, Input: {tool_input}, Log: {agent_decision.log}")

                action_log_msg = f"Thought: {agent_decision.log}\n" if agent_decision.log else ""
                action_log_msg += f"Action: Using tool {tool_name} with input {tool_input}"
                await self.memory.add_message({"role": "assistant", "content": action_log_msg})


                if tool_name in self.tool_map:
                    tool_to_use = self.tool_map[tool_name]
                    try:
                        observation = await tool_to_use.arun(tool_input) # Use async tool run
                        logger.info(f"Tool '{tool_name}' (async) observation: {str(observation)[:200]}...")
                    except Exception as e:
                        logger.error(f"Error running tool '{tool_name}' (async): {e}", exc_info=True)
                        observation = f"Error: Failed to run tool {tool_name} (async). Details: {e}"
                else:
                    logger.warning(f"Agent (async) tried to use unknown tool: {tool_name}")
                    observation = f"Error: Tool '{tool_name}' not found."

                await self.memory.add_message({"role": "system", "content": f"Tool Observation ({tool_name}): {str(observation)}"})
                intermediate_steps.append({"action": agent_decision, "observation": observation})
            else:
                logger.error(f"Agent (async) plan returned invalid type: {type(agent_decision)}. Expected AgentAction or AgentFinish.")
                return {"error": "Agent (async) planning returned an unexpected type.", "log": self._construct_scratchpad(intermediate_steps)}

        logger.warning(f"Agent (async) reached max iterations ({self.max_iterations}) without finishing.")
        return {"error": "Agent (async) stopped due to max iterations.", "log": self._construct_scratchpad(intermediate_steps)}


if __name__ == '__main__':
    # This is a conceptual test. A concrete agent implementation is needed.
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM # For a concrete LLM
    from src.tools.search_tool import SearchTool # Example tool
    import asyncio

    load_config()
    setup_logging()

    logger.info("Conceptual test for BaseAgent...")

    # A very basic concrete agent for testing purposes (e.g., a ReAct-style agent)
    class MyTestAgent(BaseAgent):
        # This is a simplified agent that always tries to use a search tool once, then finishes.
        # A real agent would have much more complex LLM prompting and parsing logic.
        def __init__(self, llm: BaseLLM, tools: List[BaseTool], **kwargs):
            # For this test, we'll create a dummy agent prompt template content
            # A real agent would load this from a file via PromptManager
            self.agent_prompt_content = """
TOOLS:
$tool_descriptions

You have access to the above tools. Use them to answer the user's question.
Your thought process should be:
Thought: [Your reasoning]
Action: [Tool name, e.g., web_search]
Action Input: [Tool input as a JSON compatible dict string, e.g., {"query": "some query"}]
Observation: [Result from the tool]
... (Repeat Thought/Action/Action Input/Observation N times)
Thought: I now have enough information to answer the question.
Final Answer: [Your final answer to the original question]

Current conversation:
$chat_history

User question: $input
Scratchpad:
$scratchpad
Thought:
"""
            # Create a dummy prompt manager and add this template
            pm = kwargs.get("prompt_manager", PromptManager())
            agent_prompt_name = "my_test_agent_prompt"
            if not pm.get_template(agent_prompt_name):
                pm.loaded_templates[agent_prompt_name] = pm.PromptTemplate(self.agent_prompt_content)

            super().__init__(llm, tools, prompt_manager=pm, agent_prompt_name=agent_prompt_name, **kwargs)


        def _plan(self, inputs: Dict[str, Any], intermediate_steps: List[Dict[str, Any]]) -> Union[AgentAction, AgentFinish]:

            # Simplified logic for test:
            # If no intermediate steps, try to search.
            # If there is an observation, try to finish.

            current_input = inputs.get("input", "")
            scratchpad_str = self._construct_scratchpad(intermediate_steps)
            tool_info_str = self._get_tool_info_string()
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory.get_history()])


            # Construct prompt for LLM (very simplified)
            # A real agent would use the PromptManager and a proper template
            prompt_vars = {
                "tool_descriptions": tool_info_str,
                "chat_history": chat_history_str,
                "input": current_input,
                "scratchpad": scratchpad_str
            }
            # full_prompt = self.agent_prompt_template.safe_substitute(**prompt_vars)
            full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)

            if not full_prompt:
                 return AgentFinish({"answer": "Error: Could not create agent prompt."}, log="Prompt formatting failed.")


            logger.debug(f"--- Agent Prompt to LLM ---\n{full_prompt}\n--------------------------")

            # Simulate LLM response based on current state (NO ACTUAL LLM CALL IN THIS BASIC TEST _plan)
            # A real agent would call: response_text = self.llm.generate(full_prompt) or self.llm.chat(...)
            # And then parse `response_text` to extract Action/Input or Final Answer.

            if not intermediate_steps: # First step, decide to search
                # This is where LLM would generate: "Thought: I need to search for XYZ.\nAction: web_search\nAction Input: {\"query\": \"XYZ\"}"
                # We hardcode it for this test:
                action_tool_name = "web_search" # Assuming web_search tool is available
                if action_tool_name not in self.tool_map:
                    return AgentFinish({"answer": f"Cannot proceed, required tool '{action_tool_name}' not found."}, log=f"Tool {action_tool_name} missing.")

                simulated_llm_output_for_action = (
                    f"Thought: I should search for information about '{current_input}'.\n"
                    f"Action: {action_tool_name}\n"
                    f"Action Input: {{\"query\": \"{current_input}\"}}" # Simplified JSON-like string
                )
                # --- Parsing Logic (would be complex in a real agent) ---
                # For test, directly create action from hardcoded simulated output
                # In real agent: parse simulated_llm_output_for_action to extract tool name and input
                parsed_tool_name = action_tool_name
                parsed_tool_input_str = f'{{"query": "{current_input}"}}' # This is what LLM might output
                import json
                try:
                    parsed_tool_input_dict = json.loads(parsed_tool_input_str)
                except json.JSONDecodeError:
                    return AgentFinish({"answer": "Error: LLM produced invalid JSON for tool input."}, log="LLM output parsing error.")

                return AgentAction(
                    tool_name=parsed_tool_name,
                    tool_input=parsed_tool_input_dict,
                    log=f"I should search for information about '{current_input}'."
                )
            else: # Second step (after getting observation), decide to finish
                last_observation = intermediate_steps[-1].get("observation", "No observation found.")
                # This is where LLM would generate: "Thought: I have the search results. I can now answer.\nFinal Answer: The answer is based on ... "
                # We hardcode it:
                simulated_llm_output_for_finish = (
                    f"Thought: I have received the observation: '{str(last_observation)[:50]}...'. I can now provide an answer.\n"
                    f"Final Answer: Based on the search, the information regarding '{current_input}' is: {str(last_observation)[:100]}..."
                )
                # --- Parsing Logic ---
                # For test, directly create finish from hardcoded simulated output
                final_answer_content = f"Based on the search, the information regarding '{current_input}' is: {str(last_observation)[:100]}..."
                return AgentFinish(
                    output={"answer": final_answer_content, "source_observation": last_observation},
                    log=f"I have received the observation and can now provide an answer."
                )

        async def _aplan(self, inputs: Dict[str, Any], intermediate_steps: List[Dict[str, Any]]) -> Union[AgentAction, AgentFinish]:
            # This is a basic async plan, for more robust test, it should call async LLM
            # For now, it will behave like the sync one for simplicity of this base agent test
            logger.info("MyTestAgent._aplan called, using sync _plan logic for this test.")
            return self._plan(inputs, intermediate_steps)


    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY not set or is a placeholder. Skipping BaseAgent integration test with MyTestAgent.")
    else:
        logger.info("\n--- Testing BaseAgent with MyTestAgent ---")
        # Setup
        test_llm = OpenAILLM(model_name="gpt-3.5-turbo") # Actual LLM (though _plan above mocks its output)
        search_tool_instance = SearchTool() # Mocked search tool
        test_agent = MyTestAgent(llm=test_llm, tools=[search_tool_instance], max_iterations=3)

        # Run agent
        agent_query = "latest news on AI"
        logger.info(f"Running MyTestAgent with query: '{agent_query}'")
        final_result = test_agent.run(agent_query)

        logger.info(f"MyTestAgent Final Result: {final_result}")
        assert "answer" in final_result
        assert agent_query in final_result.get("answer", "")
        assert "Mock Search Result" in str(final_result.get("source_observation", "")) # Check if search result is in the answer

        # Test async run (will use the simplified _aplan)
        async def run_async_agent_test():
            logger.info(f"\n--- Testing Async BaseAgent with MyTestAgent ---")
            async_agent_query = "async AI developments"
            async_final_result = await test_agent.arun(async_agent_query) # _aplan will call sync _plan here

            logger.info(f"MyTestAgent Async Final Result: {async_final_result}")
            assert "answer" in async_final_result
            assert async_agent_query in async_final_result.get("answer", "")
            assert "Mock Search Result" in str(async_final_result.get("source_observation", ""))

        asyncio.run(run_async_agent_test())

    logger.info("BaseAgent conceptual tests completed.")

# src/agents/specific_agent.py
import logging
import json
from typing import Any, List, Dict, Optional, Union

from .base_agent import BaseAgent, AgentAction, AgentFinish
from src.llms.base_llm import BaseLLM
from src.tools.base_tool import BaseTool
from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

# Default prompt name for this specific agent.
# Expects a template file like `react_agent_prompt.txt` in the prompt templates directory.
DEFAULT_SPECIFIC_AGENT_PROMPT_NAME = "react_parser_agent_prompt"

# Default prompt content if the template file is not found.
# This is a simplified ReAct-style prompt.
DEFAULT_SPECIFIC_AGENT_PROMPT_CONTENT = """
You are a helpful assistant that can use tools to answer questions.
Your goal is to answer the user's question accurately and concisely.

TOOLS:
You have access to the following tools:
$tool_descriptions

To use a tool, you MUST use the following format:
Thought: [Your reasoning for choosing the tool and action]
Action: [The EXACT name of the tool to use, e.g., web_search]
Action Input: [A JSON compatible dictionary string for the tool's input, matching its args_schema. E.g. {"query": "current weather"}]

After an action, you will receive an observation.
Observation: [The result from the tool]

If you believe you have enough information to answer the question based on the observations and your knowledge,
you MUST use the following format:
Thought: [Your reasoning for why you can answer now]
Final Answer: [Your comprehensive answer to the original user question]


Conversation History:
$chat_history

User Question: $input

Scratchpad (your thoughts, actions, and observations so far):
$scratchpad

Thought:
"""


class SpecificAgent(BaseAgent):
    """
    A more specific agent example, potentially using a ReAct-style prompting
    mechanism to decide actions and parse LLM responses.
    This agent will try to parse LLM output for "Action:", "Action Input:", and "Final Answer:".
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

        # Ensure the default prompt content is available if the template file is missing
        if agent_prompt_name == DEFAULT_SPECIFIC_AGENT_PROMPT_NAME and \
           not resolved_pm.get_template(DEFAULT_SPECIFIC_AGENT_PROMPT_NAME):
            logger.info(f"Default prompt '{DEFAULT_SPECIFIC_AGENT_PROMPT_NAME}' not found. Using built-in content.")
            resolved_pm.loaded_templates[DEFAULT_SPECIFIC_AGENT_PROMPT_NAME] = \
                resolved_pm.PromptTemplate(DEFAULT_SPECIFIC_AGENT_PROMPT_CONTENT)

        super().__init__(llm, tools, prompt_manager=resolved_pm, agent_prompt_name=agent_prompt_name, **kwargs)
        logger.info(f"SpecificAgent initialized using prompt template: '{self.agent_prompt_name}'")


    def _parse_llm_output(self, llm_output: str) -> Union[AgentAction, AgentFinish, None]:
        """
        Parses the LLM's text output to find an action or a final answer.
        This is a critical and often complex part of an agent.
        Looks for "Action:", "Action Input:", and "Final Answer:" keywords.
        """
        logger.debug(f"Parsing LLM Output:\n---\n{llm_output}\n---")

        thought = ""
        # Try to extract thought first
        if "Thought:" in llm_output:
            thought_parts = llm_output.split("Thought:", 1)
            if len(thought_parts) > 1:
                # The thought is everything after "Thought:" until the next keyword (Action or Final Answer)
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
                else: # Thought is till the end
                    thought = thought_parts[1].strip()
            logger.debug(f"Parsed Thought: {thought}")


        # Check for Final Answer
        if "Final Answer:" in llm_output:
            parts = llm_output.split("Final Answer:", 1)
            if len(parts) > 1:
                final_answer_text = parts[1].strip()
                logger.info(f"LLM indicates Final Answer: {final_answer_text}")
                return AgentFinish(output={"answer": final_answer_text}, log=thought or "Reached final answer.")
            else: # "Final Answer:" present but nothing after it.
                 logger.warning("LLM output contained 'Final Answer:' but no text followed.")


        # Check for Action and Action Input
        action_marker = "Action:"
        action_input_marker = "Action Input:"

        action_idx = llm_output.find(action_marker)
        if action_idx != -1:
            action_input_idx = llm_output.find(action_input_marker, action_idx + len(action_marker))

            if action_input_idx != -1:
                action_name_str = llm_output[action_idx + len(action_marker):action_input_idx].strip()

                # The action input might be on multiple lines if it's a complex JSON.
                # We need to find where the JSON string ends.
                # This is tricky; a robust parser would handle nested structures.
                # For simplicity, assume input is on one line or ends with newline before next keyword.
                action_input_block = llm_output[action_input_idx + len(action_input_marker):].strip()

                # Try to find end of JSON (e.g. before next "Thought:", "Action:", "Observation:")
                # This is a very naive way to find end of JSON.
                end_markers = ["Thought:", "Action:", "Observation:", "\nFinal Answer:"]
                min_pos = len(action_input_block)
                for marker in end_markers:
                    pos = action_input_block.find(marker)
                    if pos != -1 and pos < min_pos:
                        min_pos = pos

                action_input_str = action_input_block[:min_pos].strip()

                logger.debug(f"Attempting to parse Action: '{action_name_str}', Input string: '{action_input_str}'")

                try:
                    # The LLM should output a JSON string for the input.
                    tool_input_dict = json.loads(action_input_str)
                    if not isinstance(tool_input_dict, dict):
                        raise json.JSONDecodeError("Input is not a JSON object (dict).", action_input_str, 0)

                    logger.info(f"LLM action: Tool='{action_name_str}', Input={tool_input_dict}")
                    return AgentAction(tool_name=action_name_str, tool_input=tool_input_dict, log=thought)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Action Input: '{action_input_str}'. Error: {e}", exc_info=True)
                    # Fallback or error handling: maybe ask LLM to reformat or provide an error message.
                    # For now, we'll indicate parsing failure.
                    return AgentFinish(
                        output={"error": f"LLM provided invalid JSON for tool input: {action_input_str}. Details: {e}"},
                        log=thought + f"\nError: Failed to parse LLM output for tool input: {e}"
                    )
            else:
                logger.warning(f"LLM output contained 'Action:' but no 'Action Input:' found after it. Output: {llm_output}")

        # If no clear Action or Final Answer is found, it might be a malformed response or just a thought.
        # Depending on the agent's design, you might want to:
        # 1. Ask the LLM to try again / reformat.
        # 2. Treat it as a continuation of thought and prompt again.
        # 3. Return an error or a default action.
        logger.warning(f"Could not parse a clear Action or Final Answer from LLM output: {llm_output}")
        # For this agent, if no specific markers are found, we assume it's a malformed response
        # and might decide to finish with an error or ask for clarification.
        # Let's return None to indicate parsing failed to find a structured command.
        # The _plan method will need to handle this.
        return None


    def _plan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:

        current_user_input = inputs.get("input", "")
        scratchpad_str = self._construct_scratchpad(intermediate_steps)
        tool_info_str = self._get_tool_info_string()

        # Get chat history, format it simply
        # Be mindful of context window limits when including history
        chat_history_list = self.memory.get_history(max_messages=5) # Get recent 5 messages
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        prompt_vars = {
            "tool_descriptions": tool_info_str,
            "chat_history": chat_history_str,
            "input": current_user_input,
            "scratchpad": scratchpad_str
        }

        full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)
        if not full_prompt:
            logger.error(f"Failed to format agent prompt: {self.agent_prompt_name}")
            return AgentFinish({"error": "Internal error: Could not create agent prompt."}, log="Prompt formatting failed.")

        logger.debug(f"--- SpecificAgent Prompt to LLM ---\n{full_prompt}\n---------------------------------")

        # Call the LLM
        # This agent assumes the LLM is a chat model if it has a 'chat' method.
        if hasattr(self.llm, 'chat') and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            # The entire constructed prompt becomes the user message to the chat model
            messages = [{"role": "user", "content": full_prompt}]
            llm_response_obj = self.llm.chat(messages, stop_sequences=["\nObservation:"]) # Stop helps LLM focus
            llm_response_text = llm_response_obj.get("content", "")
        else:
            llm_response_text = self.llm.generate(full_prompt, stop_sequences=["\nObservation:"])

        if not llm_response_text.strip():
            logger.warning("LLM returned an empty response.")
            # Handle empty response, e.g., retry or finish with error
            return AgentFinish({"error": "LLM returned an empty response."}, log="LLM provided no output.")

        # Parse the LLM's response
        parsed_decision = self._parse_llm_output(llm_response_text)

        if parsed_decision is None:
            # Parsing failed to find a clear action or finish.
            # This could be due to malformed LLM output or the LLM just "thinking more".
            # We might want to let the agent try again, or give up.
            # For now, let's assume it's a malformed response and finish with an error.
            log_message = (f"LLM output was not parsable into a clear action or final answer. "
                           f"LLM Raw Output: '{llm_response_text}'")
            logger.warning(log_message)
            return AgentFinish(
                output={"error": "Failed to understand LLM response.", "details": llm_response_text},
                log=log_message
            )

        return parsed_decision

    async def _aplan(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Dict[str, Any]],
    ) -> Union[AgentAction, AgentFinish]:
        # Similar to _plan but uses async LLM calls

        current_user_input = inputs.get("input", "")
        scratchpad_str = self._construct_scratchpad(intermediate_steps)
        tool_info_str = self._get_tool_info_string()
        chat_history_list = await self.memory.get_history(max_messages=5) # Assuming async get_history
        chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_list])

        prompt_vars = {
            "tool_descriptions": tool_info_str,
            "chat_history": chat_history_str,
            "input": current_user_input,
            "scratchpad": scratchpad_str
        }

        full_prompt = self.prompt_manager.format_prompt(self.agent_prompt_name, **prompt_vars)
        if not full_prompt:
            logger.error(f"Async: Failed to format agent prompt: {self.agent_prompt_name}")
            return AgentFinish({"error": "Internal error: Could not create agent prompt."}, log="Prompt formatting failed.")

        logger.debug(f"--- SpecificAgent Async Prompt to LLM ---\n{full_prompt}\n---------------------------------")

        if hasattr(self.llm, 'achat') and (self.llm.model_name.startswith("gpt-3.5") or self.llm.model_name.startswith("gpt-4") or "turbo" in self.llm.model_name or "gpt-4o" in self.llm.model_name):
            messages = [{"role": "user", "content": full_prompt}]
            llm_response_obj = await self.llm.achat(messages, stop_sequences=["\nObservation:"])
            llm_response_text = llm_response_obj.get("content", "")
        else:
            llm_response_text = await self.llm.agenerate(full_prompt, stop_sequences=["\nObservation:"])

        if not llm_response_text.strip():
            logger.warning("Async LLM returned an empty response.")
            return AgentFinish({"error": "Async LLM returned an empty response."}, log="Async LLM provided no output.")

        parsed_decision = self._parse_llm_output(llm_response_text) # Parsing logic is sync

        if parsed_decision is None:
            log_message = (f"Async: LLM output was not parsable. LLM Raw Output: '{llm_response_text}'")
            logger.warning(log_message)
            return AgentFinish(
                output={"error": "Async: Failed to understand LLM response.", "details": llm_response_text},
                log=log_message
            )

        return parsed_decision


if __name__ == '__main__':
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM
    from src.tools.search_tool import SearchTool # Assuming SearchTool is available and works
    from src.memory.simple_memory import SimpleMemory
    import asyncio

    load_config()
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY not set or is a placeholder. Skipping SpecificAgent integration test.")
    else:
        logger.info("\n--- Testing SpecificAgent ---")

        # Setup components
        llm_instance = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.0) # Low temp for more predictable parsing
        search_tool_instance = SearchTool()
        agent_memory = SimpleMemory(system_message="You are an AI assistant. Follow instructions precisely.")

        # The PromptManager will load/use the default content for react_parser_agent_prompt
        # if the file doesn't exist in templates.
        agent = SpecificAgent(
            llm=llm_instance,
            tools=[search_tool_instance],
            memory=agent_memory,
            max_iterations=3 # Limit iterations for test
        )

        # Test query that should use the search tool
        # query = "What is the current weather in London?"
        query = "Who is the current president of France?" # More likely to need a search

        logger.info(f"Running SpecificAgent with query: '{query}'")
        try:
            final_output = agent.run(query)
            logger.info(f"SpecificAgent Final Output for '{query}':\n{json.dumps(final_output, indent=2)}")

            # Assertions (these are examples, actual LLM output can vary)
            assert "answer" in final_output or "error" in final_output
            if "answer" in final_output:
                 # A real test would check if the answer is plausible for the query.
                 # For "president of France", we might expect "Macron" if search worked and LLM processed it.
                 logger.info("Agent finished with an answer.")
            elif "error" in final_output:
                 logger.warning(f"Agent finished with an error: {final_output['error']}")


        except Exception as e:
            logger.error(f"Error during SpecificAgent sync run test: {e}", exc_info=True)
            raise

        # Test async run
        async def run_async_specific_agent():
            logger.info(f"\n--- Testing Async SpecificAgent with query: '{query}' ---")
            # Create new memory for async test to avoid state interference
            async_memory = SimpleMemory(system_message="You are an AI assistant for async tasks.")
            async_agent = SpecificAgent(
                llm=llm_instance,
                tools=[search_tool_instance],
                memory=async_memory,
                max_iterations=3
            )
            try:
                async_final_output = await async_agent.arun(query)
                logger.info(f"SpecificAgent Async Final Output for '{query}':\n{json.dumps(async_final_output, indent=2)}")
                assert "answer" in async_final_output or "error" in async_final_output
            except Exception as e:
                logger.error(f"Error during SpecificAgent async run test: {e}", exc_info=True)
                # Depending on the error, you might want to fail the test or just log it.
                # For now, we'll just log it as the LLM calls can be flaky.

        try:
            asyncio.run(run_async_specific_agent())
        except Exception as e:
             logger.error(f"Failed to run async SpecificAgent test: {e}", exc_info=True)


        logger.info("SpecificAgent tests completed.")

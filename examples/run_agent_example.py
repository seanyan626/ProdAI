# examples/run_agent_example.py
import logging
import json

# Adjust imports based on your actual project structure and where config/logging setup is
# This assumes 'configs' and 'src' are sibling directories or Python path is set up.
try:
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM
    from src.agents.specific_agent import SpecificAgent # Using the more concrete agent
    from src.tools.search_tool import SearchTool
    from src.memory.simple_memory import SimpleMemory
    from src.prompts.prompt_manager import PromptManager
except ImportError:
    print("Error: Could not import necessary modules. ")
    print("Ensure your PYTHONPATH is set correctly or run this example from the project root directory.")
    print("Example: python -m examples.run_agent_example")
    exit(1)

# --- Setup ---
# Load .env file variables and configure logging
load_config()
setup_logging() # Call this early

logger = logging.getLogger("run_agent_example")

def run_example():
    """
    Runs a simple example of initializing and using an agent.
    """
    logger.info("--- Starting Agent Example ---")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("OpenAI API key is not configured in .env file or is a placeholder.")
        logger.error("Please set your OPENAI_API_KEY in the .env file to run this example.")
        return

    # 1. Initialize components
    try:
        llm = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.1) # Use a capable chat model, low temp

        # Tools (example: search tool)
        search_tool = SearchTool()
        tools = [search_tool]
        tool_names = [tool.name for tool in tools]
        logger.info(f"Agent will have access to tools: {tool_names}")

        # Memory
        memory = SimpleMemory(system_message="You are an AI assistant. Your goal is to answer the user's question.")

        # Prompt Manager (optional, SpecificAgent creates one if None)
        # pm = PromptManager() # You can customize template directory if needed

        # Agent
        # The SpecificAgent uses "react_parser_agent_prompt" by default.
        # Ensure this template exists or its default content is used.
        agent = SpecificAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            # prompt_manager=pm, # Optional
            max_iterations=5 # Set a reasonable limit for iterations
        )
        logger.info(f"SpecificAgent initialized with LLM: {llm.model_name}")

    except Exception as e:
        logger.error(f"Error during component initialization: {e}", exc_info=True)
        return

    # 2. Define a query for the agent
    # query = "What were the key highlights of the last G7 summit?"
    # query = "Can you find out who won the last FIFA world cup and what the score was?"
    query = "What is the capital of Japan and what is its current population according to a web search?"


    logger.info(f"Running agent with query: \"{query}\"")

    # 3. Run the agent
    try:
        final_result = agent.run(query) # agent.run takes string or dict
    except Exception as e:
        logger.error(f"An error occurred while running the agent: {e}", exc_info=True)
        return

    # 4. Print the result
    logger.info("--- Agent Run Complete ---")
    logger.info(f"Final Agent Output for query \"{query}\":")

    # Pretty print the JSON output
    try:
        pretty_result = json.dumps(final_result, indent=2, ensure_ascii=False)
        print(pretty_result) # Using print for direct console output of the result
    except Exception as e:
        logger.error(f"Could not serialize agent output to JSON: {e}")
        logger.info(f"Raw final agent output: {final_result}")


    # Example of accessing parts of the result:
    if isinstance(final_result, dict):
        if "answer" in final_result:
            logger.info(f"\nExtracted Answer: {final_result['answer']}")
        elif "error" in final_result:
            logger.warning(f"\nAgent finished with an error: {final_result['error']}")

        if "log" in final_result and final_result["log"]: # From AgentFinish
             logger.info(f"\nFinal Agent Log/Thought:\n{final_result['log']}")


    # You can also inspect the memory
    logger.info("\n--- Conversation History from Agent Memory ---")
    history = memory.get_history()
    for i, msg in enumerate(history):
        logger.info(f"{i+1}. Role: {msg['role']}, Content: {str(msg['content'])[:150]}...")
    logger.info("--- End of Agent Example ---")


if __name__ == "__main__":
    run_example()

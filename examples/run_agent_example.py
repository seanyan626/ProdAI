# examples/run_agent_example.py
# 运行 Agent 的示例脚本
import json
import logging

# 根据你的实际项目结构和配置/日志设置的位置调整导入
# 这里假设 'configs' 和 'core' 是同级目录，或者 Python 路径已正确设置。
try:
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from core.models.llm.openai_llm import OpenAILLM  # 更新路径和类名
    from core.agents.specific_agent import SpecificAgent
    from core.tools.search_tool import SearchTool
    from core.memory.simple_memory import SimpleMemory
    # from core.prompts.prompt_manager import PromptManager # 如果 SpecificAgent 未自动创建，则可能需要
except ImportError as e:
    print(f"错误: 无法导入必要的模块: {e}")
    print("请确保你的 PYTHONPATH 设置正确，或者从项目根目录运行此示例。")
    print("例如: python -m examples.run_agent_example")
    exit(1)

# --- 设置 ---
# 加载 .env 文件变量并配置日志
load_config()
setup_logging()  # 尽早调用

logger = logging.getLogger("run_agent_example_脚本")


def run_example():
    """
    运行一个初始化和使用 Agent 的简单示例。
    用户需要先在 src/ 目录下实现相关模块的具体逻辑。
    """
    logger.info("--- 开始 Agent 示例 ---")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.error("OpenAI API 密钥未在 .env 文件中配置或仍为占位符。")
        logger.error("请在 .env 文件中设置你的 OPENAI_API_KEY 以便运行此示例。")
        return  # 对于需要API密钥的真实LLM调用，这里应该返回

    # 1. 初始化组件
    agent = None  # 初始化 agent 变量
    try:
        logger.info("尝试初始化组件...")
        # 注意：OpenAILLM 现在内部使用 langchain
        llm = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.1)  # <--- 已更改类名回 OpenAILLM
        logger.info(f"LLM ({getattr(llm, 'model_name', '未知模型')}) 初始化完成。")  # <--- "语言模型" -> "LLM"

        search_tool = SearchTool()
        tools = [search_tool]
        tool_names = [getattr(tool, 'name', '未知工具') for tool in tools]
        logger.info(f"Agent 将有权访问以下工具: {tool_names}")

        memory = SimpleMemory(system_message="你是一个 AI 助手。你的目标是回答用户的问题。")
        logger.info("记忆模块初始化完成。")

        # SpecificAgent 默认使用 "react_parser_agent_prompt" 模板。
        # 它会在内部创建 PromptManager (如果未提供)。
        agent = SpecificAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            max_iterations=5
        )
        logger.info(f"SpecificAgent 使用 LLM '{llm.model_name}' 初始化完成。")

    except Exception as e:
        logger.error(f"组件初始化期间出错: {e}", exc_info=True)
        return

    # 2. 为 Agent 定义一个查询
    query = "日本的首都是哪里？以及它当前的人口大约是多少（请通过网络搜索）？"  # 更明确地指示使用搜索
    logger.info(f"定义查询: \"{query}\"")

    # 3. 运行 Agent
    final_result = None
    if agent:  # 确保 agent 已成功初始化
        logger.info(f"正在使用 Agent ({agent.__class__.__name__}) 运行查询...")
        try:
            final_result = agent.run(query)
        except Exception as e:
            logger.error(f"运行 Agent 时发生错误: {e}", exc_info=True)
            final_result = {"error": f"Agent 运行失败: {e}"}  # 设置错误结果以便后续处理
    else:
        logger.error("Agent 未能成功初始化，无法运行查询。")
        final_result = {"error": "Agent 初始化失败。"}

    # 4. 打印结果
    logger.info("--- Agent 运行完成 ---")
    logger.info(f"对于查询 \"{query}\" 的 Agent 最终输出:")

    try:
        pretty_result = json.dumps(final_result, indent=2, ensure_ascii=False)
        print(pretty_result)
    except Exception as e:
        logger.error(f"无法将 Agent 输出序列化为 JSON: {e}")
        logger.info(f"原始 Agent 最终输出: {final_result}")

    if isinstance(final_result, dict):
        if "answer" in final_result:
            logger.info(f"\n提取的答案: {final_result['answer']}")
        elif "error" in final_result:
            logger.warning(f"\nAgent 完成但出现错误: {final_result['error']}")

        agent_log = final_result.get("log")  # AgentFinish 可能包含log
        if agent_log:
            logger.info(f"\nAgent 最终日志/思考:\n{agent_log}")

    if agent and hasattr(agent, 'memory'):
        logger.info("\n--- 来自 Agent 内存的对话历史 ---")
        history = agent.memory.get_history()
        if history:
            for i, msg in enumerate(history):
                logger.info(f"{i + 1}. 角色: {msg.get('role')}, 内容: {str(msg.get('content'))[:150]}...")
        else:
            logger.info("内存中没有历史记录。")

    logger.info("--- Agent 示例结束 ---")


if __name__ == "__main__":
    run_example()

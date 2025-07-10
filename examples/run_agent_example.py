# examples/run_agent_example.py
# 运行 Agent 的示例脚本
import logging
import json

# 根据你的实际项目结构和配置/日志设置的位置调整导入
# 这里假设 'configs' 和 'src' 是同级目录，或者 Python 路径已正确设置。
try:
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    from src.llms.openai_llm import OpenAILLM
    from src.agents.specific_agent import SpecificAgent # 使用更具体的 Agent 实现
    from src.tools.search_tool import SearchTool
    from src.memory.simple_memory import SimpleMemory
    # from src.prompts.prompt_manager import PromptManager # 如果需要自定义 PromptManager 实例
except ImportError:
    print("错误: 无法导入必要的模块。")
    print("请确保你的 PYTHONPATH 设置正确，或者从项目根目录运行此示例。")
    print("例如: python -m examples.run_agent_example")
    exit(1)

# --- 设置 ---
# 加载 .env 文件变量并配置日志
load_config()
setup_logging() # 尽早调用

logger = logging.getLogger("run_agent_example_脚本") # 为此示例脚本设置特定logger名称

def run_example():
    """
    运行一个初始化和使用 Agent 的简单示例。
    """
    logger.info("--- 开始 Agent 示例 ---")

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE": # 检查 OpenAI API 密钥
        logger.error("OpenAI API 密钥未在 .env 文件中配置或仍为占位符。")
        logger.error("请在 .env 文件中设置你的 OPENAI_API_KEY 以运行此示例。")
        return

    # 1. 初始化组件
    try:
        llm = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.1) # 使用一个强大的聊天模型，低温模式

        # 工具 (示例: 搜索工具)
        search_tool = SearchTool()
        tools = [search_tool]
        tool_names = [tool.name for tool in tools]
        logger.info(f"Agent 将有权访问以下工具: {tool_names}")

        # 记忆
        memory = SimpleMemory(system_message="你是一个 AI 助手。你的目标是回答用户的问题。") # 中文系统消息

        # 提示管理器 (可选, 如果为 None，SpecificAgent 会创建一个)
        # pm = PromptManager() # 如果需要，你可以自定义模板目录

        # Agent
        # SpecificAgent 默认使用 "react_parser_agent_prompt"。
        # 确保此模板存在或其默认内容被使用。
        agent = SpecificAgent(
            llm=llm,
            tools=tools,
            memory=memory,
            # prompt_manager=pm, # 可选
            max_iterations=5 # 为迭代次数设置一个合理的限制
        )
        logger.info(f"SpecificAgent 已使用 LLM: {llm.model_name} 初始化。")

    except Exception as e:
        logger.error(f"组件初始化期间出错: {e}", exc_info=True)
        return

    # 2. 为 Agent 定义一个查询
    # query = "上次G7峰会的主要亮点是什么？"
    # query = "你能查出上一届FIFA世界杯谁赢了，比分是多少吗？"
    query = "日本的首都是哪里？根据网络搜索，它目前的人口是多少？"


    logger.info(f"正在使用查询运行 Agent: \"{query}\"")

    # 3. 运行 Agent
    try:
        final_result = agent.run(query) # agent.run 接受字符串或字典
    except Exception as e:
        logger.error(f"运行 Agent 时发生错误: {e}", exc_info=True)
        return

    # 4. 打印结果
    logger.info("--- Agent 运行完成 ---")
    logger.info(f"对于查询 \"{query}\" 的 Agent 最终输出:")

    # JSON 格式美化输出
    try:
        pretty_result = json.dumps(final_result, indent=2, ensure_ascii=False) # ensure_ascii=False 以正确显示中文
        print(pretty_result) # 使用 print 直接在控制台输出结果
    except Exception as e:
        logger.error(f"无法将 Agent 输出序列化为 JSON: {e}")
        logger.info(f"原始 Agent 最终输出: {final_result}")


    # 访问结果部分的示例:
    if isinstance(final_result, dict):
        if "answer" in final_result:
            logger.info(f"\n提取的答案: {final_result['answer']}")
        elif "error" in final_result:
            logger.warning(f"\nAgent 完成但出现错误: {final_result['error']}")

        if "log" in final_result and final_result["log"]: # 来自 AgentFinish
             logger.info(f"\nAgent 最终日志/思考:\n{final_result['log']}")


    # 你也可以检查内存
    logger.info("\n--- 来自 Agent 内存的对话历史 ---")
    history = memory.get_history()
    for i, msg in enumerate(history):
        logger.info(f"{i+1}. 角色: {msg['role']}, 内容: {str(msg['content'])[:150]}...")
    logger.info("--- Agent 示例结束 ---")


if __name__ == "__main__":
    run_example()

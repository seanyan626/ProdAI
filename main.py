# main.py
# 项目主入口文件
import logging
from configs.config import load_config, APP_NAME
from configs.logging_config import setup_logging

# 加载应用配置
load_config()

# 设置日志
setup_logging()

logger = logging.getLogger(__name__)

def main():
    """
    运行 AI 项目的主函数。
    这是一个占位符，应根据项目的特定入口点进行扩展。
    例如，它可以启动一个命令行界面 (CLI)、一个 Web 服务器或一个 Agent 交互循环。

    注意：由于核心模块代码已被简化为骨架，以下示例代码已被注释掉。
    用户需要先在 core/ 目录下实现相关模块的具体逻辑后才能取消注释并运行。
    """
    logger.info(f"正在启动 {APP_NAME}...")
    logger.info("这是一个骨架项目。请在 core/ 目录下的模块中添加您的代码实现。") # src -> core

    # 示例：初始化并运行一个 Agent (实际实现将取决于你的 Agent 设置)
    # try:
    #     # --- 以下代码块依赖于 core/ 目录下的具体实现 ---
    #     # from core.agents.specific_agent import SpecificAgent
    #     # from core.models.language.openai_language_model import OpenAILanguageModel # 更新路径和类名
    #     # from core.tools.search_tool import SearchTool
    #     # from core.memory.simple_memory import SimpleMemory

    #     # logger.info("尝试初始化组件...")
    #     # llm = OpenAILanguageModel() # 更新类名
    #     # logger.info("语言模型初始化完成 (模拟)。") # LLM -> 语言模型

    #     # search = SearchTool()
    #     # tools = [search]
    #     # logger.info(f"工具列表: {[t.name for t in tools]} (模拟)")

    #     # memory = SimpleMemory(system_message="你是一个AI助手。")
    #     # logger.info("记忆模块初始化完成 (模拟)。")

    #     # agent = SpecificAgent(llm=llm, tools=tools, memory=memory)
    #     # logger.info("Agent 初始化完成 (模拟)。")

    #     # logger.info("尝试运行 Agent...")
    #     # query = "日本的首都是哪里？"
    #     # result = agent.run(query)
    #     # logger.info(f"Agent 对 '{query}' 的响应: {result}")

    #     logger.info("主要应用逻辑占位符。") # 如果没有实际Agent运行，则打印此信息

    # except ImportError as e:
    #     logger.error(f"导入模块失败，请确保 core/ 目录下的模块已实现: {e}") # src -> core
    # except Exception as e:
    #     logger.error(f"应用执行期间发生错误: {e}", exc_info=True)


    # --- 模型直接调用测试 ---
    # 您可以取消注释以下代码块之一来直接测试特定的LLM实现。
    # 请确保您已在 .env 文件中配置了相应的 API 密钥。

    # --- 测试 OpenAI 模型 ---
    try:
        from core.models.llm.openai_llm import OpenAILLM
        from configs.config import OPENAI_API_KEY # 再次导入以确保检查准确
        if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_API_KEY_HERE":
            logger.info("\n--- 正在测试 OpenAILLM ---")
            openai_llm = OpenAILLM()
            messages = [{"role": "user", "content": "你好，请用英文介绍一下你自己。"}]
            response = openai_llm.chat(messages)
            logger.info(f"OpenAI LLM 响应: {response.get('content')}")
        else:
            logger.warning("未配置 OpenAI API 密钥，跳过 OpenAI LLM 测试。")
    except ImportError:
        logger.warning("无法导入 OpenAILLM，跳过测试。请确保该模块已实现。")
    except Exception as e:
        logger.error(f"测试 OpenAILLM 时出错: {e}", exc_info=True)


    # except Exception as e:
    #     logger.error(f"测试 DashScopeLLM 时出错: {e}", exc_info=True)


    # --- 测试 DeepSeek 模型 ---
    try:
        from core.models.llm.deepseek_llm import DeepSeekLLM
        from configs.config import DEEPSEEK_API_KEY
        if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_API_KEY_HERE":
            logger.info("\n--- 正在测试 DeepSeekLLM ---")
            deepseek_llm = DeepSeekLLM()
            messages = [{"role": "user", "content": "你好，请用中文介绍一下你自己，说明你是哪个公司的模型。"}]
            response = deepseek_llm.chat(messages)
            logger.info(f"DeepSeek LLM 响应: {response.get('content')}")
        else:
            logger.warning("未配置 DeepSeek API 密钥，跳过 DeepSeek LLM 测试。")
    except ImportError:
        logger.warning("无法导入 DeepSeekLLM，跳过测试。请确保该模块已实现。")
    except Exception as e:
        logger.error(f"测试 DeepSeekLLM 时出错: {e}", exc_info=True)


    logger.info(f"{APP_NAME} (骨架) 已启动并结束。请添加您的实现。")

if __name__ == "__main__":
    main()

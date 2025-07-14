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
    """
    logger.info(f"正在启动 {APP_NAME}...")

    # --- 模型直接调用测试 ---
    # 您可以取消注释以下代码块之一来直接测试特定的LLM实现。
    # 请确保您已在 .env 文件中配置了相应的 API 密钥。

    # --- 测试 OpenAI 模型 ---
    # try:
    #     from core.models.llm.openai_llm import OpenAILLM
    #     from configs.config import OPENAI_API_KEY
    #     if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_KEY_HERE":
    #         logger.info("\n--- 正在测试 OpenAILLM ---")
    #         openai_llm = OpenAILLM()
    #         messages = [{"role": "user", "content": "Hello, introduce yourself in English."}]
    #         response = openai_llm.chat(messages)
    #         logger.info(f"OpenAI LLM 响应: {response.get('content')}")
    #     else:
    #         logger.warning("未配置 OpenAI API 密钥，跳过 OpenAI LLM 测试。")
    # except ImportError:
    #     logger.warning("无法导入 OpenAILLM，跳过测试。请确保该模块已实现。")
    # except Exception as e:
    #     logger.error(f"测试 OpenAILLM 时出错: {e}", exc_info=True)


    # --- 测试 DashScope 模型 ---
    try:
        from core.models.llm.dashscope_llm import DashScopeLLM
        from configs.config import DASHSCOPE_API_KEY
        if DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "YOUR_DASHSCOPE_API_KEY_HERE":
            logger.info("\n--- 正在测试 DashScopeLLM (Langchain 封装版) ---")
            dashscope_llm = DashScopeLLM(model_name="qwen-turbo")
            messages = [{"role": "user", "content": "你好，介绍一下你自己，说明你是哪个模型。"}]
            response = dashscope_llm.chat(messages)
            logger.info(f"DashScope LLM 响应: {response.get('content')}")
        else:
            logger.warning("未配置 DashScope API 密钥，跳过 DashScope LLM 测试。")
    except ImportError:
        logger.warning("无法导入 DashScopeLLM，跳过测试。请确保该模块已实现。")
    except Exception as e:
        logger.error(f"测试 DashScopeLLM 时出错: {e}", exc_info=True)


    # --- 测试 DeepSeek 模型 ---
    # try:
    #     from core.models.llm.deepseek_llm import DeepSeekLLM
    #     from configs.config import DEEPSEEK_API_KEY
    #     if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_API_KEY_HERE":
    #         logger.info("\n--- 正在测试 DeepSeekLLM ---")
    #         deepseek_llm = DeepSeekLLM()
    #         messages = [{"role": "user", "content": "你好，请用中文介绍一下你自己，说明你是哪个公司的模型。"}]
    #         response = deepseek_llm.chat(messages)
    #         logger.info(f"DeepSeek LLM 响应: {response.get('content')}")
    #     else:
    #         logger.warning("未配置 DeepSeek API 密钥，跳过 DeepSeek LLM 测试。")
    # except ImportError:
    #     logger.warning("无法导入 DeepSeekLLM，跳过测试。请确保该模块已实现。")
    # except Exception as e:
    #     logger.error(f"测试 DeepSeekLLM 时出错: {e}", exc_info=True)


    logger.info(f"{APP_NAME} 已结束。")

if __name__ == "__main__":
    main()

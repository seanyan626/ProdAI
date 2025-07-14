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
    此文件包含用于直接测试各个模型实现的示例代码块。

    """
    logger.info(f"正在启动 {APP_NAME}...")

    # --- 模型直接调用测试 ---

    # 使用方法：
    # 1. 在 .env 文件中配置好您想测试的模型的 API Key (和 URL/BASE, 如果需要)。
    # 2. 取消下面对应模型测试代码块的注释。
    # 3. 运行 `python main.py`。
    # 4. 建议一次只测试一个模型。

    # --- 测试 OpenAI LLM ---

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

    #     logger.warning("无法导入 OpenAILLM，跳过测试。")

    # except Exception as e:
    #     logger.error(f"测试 OpenAILLM 时出错: {e}", exc_info=True)

    # --- 测试 DashScope LLM ---
    try:
        from core.models.llm.dashscope_llm import DashScopeLLM
        from configs.config import DASHSCOPE_API_KEY
        if DASHSCOPE_API_KEY and DASHSCOPE_API_KEY != "YOUR_DASHSCOPE_API_KEY_HERE":
            logger.info("\n--- 正在测试 DashScopeLLM ---")
            dashscope_llm = DashScopeLLM(model_name="qwen-turbo")
            messages = [{"role": "user", "content": "你好，介绍一下你自己，说明你是哪个模型。"}]
            response = dashscope_llm.chat(messages)
            logger.info(f"DashScope LLM 响应: {response.get('content')}")
        else:
            logger.warning("未配置 DashScope API 密钥，跳过 DashScope LLM 测试。")
    except ImportError:
        logger.warning("无法导入 DashScopeLLM，跳过测试。")
    except Exception as e:
        logger.error(f"测试 DashScopeLLM 时出错: {e}", exc_info=True)

    # --- 测试 DeepSeek LLM ---

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

    #     logger.warning("无法导入 DeepSeekLLM，跳过测试。")

    # except Exception as e:
    #     logger.error(f"测试 DeepSeekLLM 时出错: {e}", exc_info=True)

    # --- 测试 OpenAI Embedding 模型 ---
    # try:
    #     from core.models.embedding.openai_embedding_model import OpenAIEmbeddingModel
    #     from configs.config import OPENAI_API_KEY
    #     if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_KEY_HERE":
    #         logger.info("\n--- 正在测试 OpenAIEmbeddingModel ---")
    #         embedding_model = OpenAIEmbeddingModel()
    #         query_text = "这是一个用于测试嵌入的查询"
    #         embedding_vector = embedding_model.embed_query(query_text)
    #         logger.info(f"查询 '{query_text}' 的嵌入向量 (前5维): {embedding_vector[:5]}")
    #         logger.info(f"向量维度: {len(embedding_vector)}")
    #     else:
    #         logger.warning("未配置 OpenAI API 密钥，跳过 OpenAI Embedding 模型测试。")
    # except ImportError:
    #     logger.warning("无法导入 OpenAIEmbeddingModel，跳过测试。")
    # except Exception as e:
    #     logger.error(f"测试 OpenAIEmbeddingModel 时出错: {e}", exc_info=True)

    logger.info(f"{APP_NAME} 运行结束。")


if __name__ == "__main__":
    main()

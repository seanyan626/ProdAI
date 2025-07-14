import logging
import os

from dotenv import load_dotenv

# --- 应用元数据 ---

APP_NAME = "AI 项目框架"
APP_VERSION = "0.1.0"

# --- 配置变量 (在此处声明，以便IDE和linter识别) ---
# API 密钥和端点
OPENAI_API_KEY: str = None
OPENAI_API_BASE: str = None
DASHSCOPE_API_KEY: str = None
DASHSCOPE_API_URL: str = None
DEEPSEEK_API_KEY: str = None
DEEPSEEK_API_URL: str = None

# LLM 设置
DEFAULT_LLM_MODEL: str = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS: int = 1500
DEFAULT_TEMPERATURE: float = 0.7

# Agent 设置
MAX_ITERATIONS: int = 10

# 日志配置
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: str = None



# --- 环境变量加载 ---
def load_env():
    """
    从 .env 文件加载环境变量。
    会在当前目录或父目录中查找 .env 文件。
    """
    # 确定基础目录（项目根目录）
    # 假设 config.py 位于项目根目录的子目录中。
    # 如果你的结构不同，请相应调整。
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # 从 configs 向上移动一级到项目根目录

    dotenv_path = os.path.join(project_root, ".env")

    if not os.path.exists(dotenv_path):
        logging.warning(
            f".env 文件未在 {dotenv_path} 找到。"
            "请确保它存在并包含必要的配置。"
            "你可以将 .env.example 复制为 .env 并填入你的详细信息。"
        )
        # 如果需要，可以回退到尝试从当前工作目录加载，
        # 但显式路径更好。
        load_dotenv()
    else:
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"已从 {dotenv_path} 加载环境变量。")


_ENV_LOADED = False  # 标记环境变量是否已加载


def load_config():
    """
    从 .env 文件加载环境变量，并更新模块级的配置变量。
    确保此函数在任何其他模块使用配置变量之前被调用一次。
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    # 1. 从 .env 文件加载环境变量
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dotenv_path = os.path.join(project_root, ".env")


    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"已从 {dotenv_path} 加载环境变量。")
    else:
        logging.warning(f".env 文件未在 {dotenv_path} 找到。将仅使用系统中已有的环境变量。")
        # 即使文件不存在，也调用一次 load_dotenv() 以加载系统级环境变量
        load_dotenv()

    # 2. 更新模块级的全局变量
    g = globals()


    # API 密钥和端点
    g['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    g['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE")
    g['DASHSCOPE_API_KEY'] = os.getenv("DASHSCOPE_API_KEY")
    g['DASHSCOPE_API_URL'] = os.getenv("DASHSCOPE_API_URL")
    g['DEEPSEEK_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
    g['DEEPSEEK_API_URL'] = os.getenv("DEEPSEEK_API_URL")

    # LLM 设置
    g['DEFAULT_LLM_MODEL'] = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    g['DEFAULT_MAX_TOKENS'] = int(os.getenv("DEFAULT_MAX_TOKENS", "1500"))
    g['DEFAULT_TEMPERATURE'] = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))


    # Agent 设置
    g['MAX_ITERATIONS'] = int(os.getenv("MAX_ITERATIONS", "10"))

    # 日志配置
    g['LOG_LEVEL'] = os.getenv("LOG_LEVEL", "INFO").upper()
    g['LOG_FORMAT'] = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    g['LOG_FILE'] = os.getenv("LOG_FILE")

    _ENV_LOADED = True
    logging.info("配置变量已加载。")


if __name__ == "__main__":
    # 这部分用于测试配置加载
    # 1. 先不加载，直接打印，值应该是模块顶部的默认值
    print("--- 加载前 ---")
    print(f"OpenAI API 密钥: {OPENAI_API_KEY}")

    # 2. 调用加载函数
    load_config()

    # 3. 再次打印，值应该是从 .env 文件或环境变量中加载的
    print("\n--- 加载后 ---")
    print("应用名称:", APP_NAME)
    print("默认 LLM 模型:", DEFAULT_LLM_MODEL)
    print("日志级别:", LOG_LEVEL)
    print("\n--- LLM API 配置加载状态 ---")
    print(f"OpenAI API 密钥: {'已加载' if OPENAI_API_KEY else '未找到'}")
    print(f"OpenAI API Base URL: {OPENAI_API_BASE or '未设置'}")
    print(f"DashScope API 密钥: {'已加载' if DASHSCOPE_API_KEY else '未找到'}")
    print(f"DashScope API URL: {DASHSCOPE_API_URL or '未设置'}")
    print(f"DeepSeek API 密钥: {'已加载' if DEEPSEEK_API_KEY else '未找到'}")
    print(f"DeepSeek API URL: {DEEPSEEK_API_URL or '未设置'}")

    # 4. 再次调用，应该不会重复加载
    print("\n--- 尝试重复加载 ---")
    load_config()

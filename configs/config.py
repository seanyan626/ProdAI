import os
import logging
from dotenv import load_dotenv

# --- 应用元数据 ---
APP_NAME = "AI 项目框架"  # 应用名称
APP_VERSION = "0.1.0"    # 应用版本

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
    project_root = os.path.dirname(current_dir) # 从 configs 向上移动一级到项目根目录

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


_ENV_LOADED = False # 标记环境变量是否已加载

def load_config():
    """
    加载所有配置。确保 .env 文件只加载一次。
    """
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_env()
        _ENV_LOADED = True

# --- API 密钥和端点 ---
# 在调用 load_config() 后加载它们
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # Pinecone API 密钥 (如果使用)
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Pinecone 环境 (如果使用)
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") # Anthropic API 密钥 (如果使用)


# --- LLM 设置 ---
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo") # 默认 LLM 模型
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1500"))   # 默认最大 token 数
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7")) # 默认温度参数

# --- Agent 设置 ---
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10")) # Agent 最大迭代次数


# --- RAG 设置 ---
# 示例: 知识库路径，可以是相对于项目根目录的路径
# KNOWLEDGE_BASE_PATH = os.path.join(os.getenv("PROJECT_ROOT_DIR", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "knowledge_base") # 知识库路径
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002") # 嵌入模型
# VECTOR_STORE_PATH = os.path.join(os.getenv("PROJECT_ROOT_DIR", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "vector_store") # 向量存储路径


# --- 日志配置 ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" # 日志格式
LOG_FILE = os.getenv("LOG_FILE", None) # 日志输出文件路径 (例如 "app.log")


# --- 其他配置 ---
# 在此添加你的应用可能需要的任何其他全局配置。


# --- 辅助函数，确保在访问变量前加载配置 ---
# 如果模块直接导入特定的配置变量，这很有用。
# 在主应用脚本（例如 main.py）的开头调用 load_config()，
# 或确保在访问任何配置变量之前调用它。

# 访问配置值的示例:
# from configs.config import OPENAI_API_KEY, load_config
# load_config() # 确保已加载
# print(OPENAI_API_KEY)

if __name__ == "__main__":
    # 这部分用于测试配置加载
    load_config()
    print(f"应用名称: {APP_NAME}")
    print(f"OpenAI API 密钥: {'已加载' if OPENAI_API_KEY else '未找到'}")
    print(f"默认 LLM 模型: {DEFAULT_LLM_MODEL}")
    print(f"日志级别: {LOG_LEVEL}")
    # print(f"知识库路径: {KNOWLEDGE_BASE_PATH}") # 如果使用，取消注释
    # project_root_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f"项目根目录 (推断): {project_root_test}")
    # print(f"正在检查的 .env 文件完整路径: {os.path.join(project_root_test, '.env')}")

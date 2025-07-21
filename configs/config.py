# configs/config.py
"""
应用配置管理模块

提供统一的配置管理，支持从环境变量和.env文件加载配置。
使用配置类来组织不同类型的配置，提供更好的结构和类型安全。
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
    """应用基础配置"""
    name: str = "AI 项目框架"
    version: str = "0.1.0"
    debug: bool = False


@dataclass
class OpenAIConfig:
    """OpenAI API 配置"""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    # OpenAI Embedding 专用配置
    embedding_api_key: Optional[str] = None
    embedding_name: str = "text-embedding-ada-002"
    embedding_api_url: Optional[str] = None


@dataclass
class DashScopeConfig:
    """DashScope API 配置 - 统一的阿里百炼配置"""
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    # LLM 模型配置
    llm_model_name: str = "qwen-plus"
    # Embedding 模型配置
    embedding_model_name: str = "text-embedding-v1"
    # OCR 模型配置
    ocr_model_name: str = "qwen-vl-max"


@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM 通用配置"""
    default_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1500
    temperature: float = 0.7


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_iterations: int = 10


@dataclass
class OCRConfig:
    """OCR 配置"""
    default_model: str = "openai_vision"
    max_retries: int = 3
    timeout: int = 30

    # 云服务 API 配置
    aliyun_api_key: Optional[str] = None
    baidu_api_key: Optional[str] = None
    baidu_secret_key: Optional[str] = None
    tencent_secret_id: Optional[str] = None
    tencent_secret_key: Optional[str] = None
    google_vision_api_key: Optional[str] = None
    azure_vision_api_key: Optional[str] = None
    azure_vision_endpoint: Optional[str] = None


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


def _load_env_file():
    """加载 .env 文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dotenv_path = os.path.join(project_root, ".env")

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"已从 {dotenv_path} 加载环境变量")
    else:
        logging.warning(f".env 文件未在 {dotenv_path} 找到，将仅使用系统环境变量")
        load_dotenv()


class Config:
    """
    统一配置管理类
    
    提供单例模式的配置管理，支持从环境变量加载配置。
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_config()
            Config._initialized = True

    def _load_config(self):
        """从环境变量加载配置"""
        # 加载 .env 文件
        _load_env_file()

        # 初始化各个配置模块
        self.app = AppConfig(
            name=os.getenv("APP_NAME", "AI 项目框架"),
            version=os.getenv("APP_VERSION", "0.1.0"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )

        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
            embedding_api_key=os.getenv("OPENAI_EMBEDDING_MODEL_API_KEY"),
            embedding_name=os.getenv("OPENAI_EMBEDDING_NAME", "text-embedding-ada-002"),
            embedding_api_url=os.getenv("OPENAI_EMBEDDING_API_URL")
        )

        self.dashscope = DashScopeConfig(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            api_url=os.getenv("DASHSCOPE_API_URL"),
            llm_model_name=os.getenv("DASHSCOPE_LLM_MODEL_NAME", "qwen-plus"),
            embedding_model_name=os.getenv("DASHSCOPE_EMBEDDING_NAME", "text-embedding-v1"),
            ocr_model_name=os.getenv("DASHSCOPE_OCR_NAME", "qwen-vl-max")
        )

        self.deepseek = DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_url=os.getenv("DEEPSEEK_API_URL"),
            model_name=os.getenv("DEEPSEEK_MODEL_NAME")
        )

        self.llm = LLMConfig(
            default_model=os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS", "1500")),
            temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        )

        self.agent = AgentConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10"))
        )

        self.ocr = OCRConfig(
            default_model=os.getenv("DEFAULT_OCR_MODEL", "openai_vision"),
            max_retries=int(os.getenv("OCR_MAX_RETRIES", "3")),
            timeout=int(os.getenv("OCR_TIMEOUT", "30")),
            aliyun_api_key=os.getenv("ALIYUN_OCR_API_KEY"),
            baidu_api_key=os.getenv("BAIDU_OCR_API_KEY"),
            baidu_secret_key=os.getenv("BAIDU_OCR_SECRET_KEY"),
            tencent_secret_id=os.getenv("TENCENT_OCR_SECRET_ID"),
            tencent_secret_key=os.getenv("TENCENT_OCR_SECRET_KEY"),
            google_vision_api_key=os.getenv("GOOGLE_VISION_API_KEY"),
            azure_vision_api_key=os.getenv("AZURE_VISION_API_KEY"),
            azure_vision_endpoint=os.getenv("AZURE_VISION_ENDPOINT")
        )

        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=os.getenv("LOG_FILE")
        )

        logging.info("配置已加载完成")

    def get_api_status(self) -> dict:
        """获取 API 配置状态"""
        return {
            "openai": {
                "api_key": "已配置" if self.openai.api_key else "未配置",
                "api_base": self.openai.api_base or "默认",
                "embedding_model": self.openai.embedding_model
            },
            "dashscope": {
                "api_key": "已配置" if self.dashscope.api_key else "未配置",
                "api_url": self.dashscope.api_url or "默认",
                "llm_model_name": self.dashscope.llm_model_name or "未设置",
                "embedding_model_name": self.dashscope.embedding_model_name or "未设置",
                "ocr_model_name": self.dashscope.ocr_model_name or "未设置"
            },
            "deepseek": {
                "api_key": "已配置" if self.deepseek.api_key else "未配置",
                "api_url": self.deepseek.api_url or "默认",
                "model_name": self.deepseek.model_name or "未设置"
            }
        }

    def get_ocr_status(self) -> dict:
        """获取 OCR 配置状态"""
        return {
            "default_model": self.ocr.default_model,
            "max_retries": self.ocr.max_retries,
            "timeout": f"{self.ocr.timeout}秒",
            "providers": {
                "aliyun": "已配置" if self.ocr.aliyun_api_key else "未配置",
                "baidu": "已配置" if self.ocr.baidu_api_key else "未配置",
                "tencent": "已配置" if self.ocr.tencent_secret_id else "未配置",
                "google": "已配置" if self.ocr.google_vision_api_key else "未配置",
                "azure": "已配置" if self.ocr.azure_vision_api_key else "未配置"
            }
        }


# 创建全局配置实例
config = Config()

# 为了向后兼容，保留原有的全局变量
# 应用元数据
APP_NAME = config.app.name
APP_VERSION = config.app.version

# OpenAI 配置
OPENAI_API_KEY = config.openai.api_key
OPENAI_API_BASE = config.openai.api_base
OPENAI_EMBEDDING_MODEL_NAME = config.openai.embedding_model

# DashScope 配置
DASHSCOPE_API_KEY = config.dashscope.api_key
DASHSCOPE_API_URL = config.dashscope.api_url
DASHSCOPE_LLM_MODEL_NAME = config.dashscope.llm_model_name
DASHSCOPE_EMBEDDING_MODEL_NAME = config.dashscope.embedding_model_name
DASHSCOPE_OCR_MODEL_NAME = config.dashscope.ocr_model_name

# DeepSeek 配置
DEEPSEEK_API_KEY = config.deepseek.api_key
DEEPSEEK_API_URL = config.deepseek.api_url
DEEPSEEK_MODEL_NAME = config.deepseek.model_name

# LLM 配置
DEFAULT_LLM_MODEL = config.llm.default_model
DEFAULT_MAX_TOKENS = config.llm.max_tokens
DEFAULT_TEMPERATURE = config.llm.temperature

# Agent 配置
MAX_ITERATIONS = config.agent.max_iterations

# OCR 配置
DEFAULT_OCR_MODEL = config.ocr.default_model
OCR_MAX_RETRIES = config.ocr.max_retries
OCR_TIMEOUT = config.ocr.timeout
ALIYUN_OCR_API_KEY = config.ocr.aliyun_api_key
BAIDU_OCR_API_KEY = config.ocr.baidu_api_key
BAIDU_OCR_SECRET_KEY = config.ocr.baidu_secret_key
TENCENT_OCR_SECRET_ID = config.ocr.tencent_secret_id
TENCENT_OCR_SECRET_KEY = config.ocr.tencent_secret_key
GOOGLE_VISION_API_KEY = config.ocr.google_vision_api_key
AZURE_VISION_API_KEY = config.ocr.azure_vision_api_key
AZURE_VISION_ENDPOINT = config.ocr.azure_vision_endpoint

# 日志配置
LOG_LEVEL = config.logging.level
LOG_FORMAT = config.logging.format
LOG_FILE = config.logging.file


def load_config():
    """
    向后兼容的配置加载函数
    
    注意：新的配置系统会自动加载，这个函数主要用于向后兼容。
    """
    global config
    if not Config._initialized:
        config = Config()
    return config


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
    print(f"DashScope LLM MODEL NAME: {DASHSCOPE_LLM_MODEL_NAME or '未设置'}")
    print(f"DashScope EMBEDDING MODEL NAME: {DASHSCOPE_EMBEDDING_MODEL_NAME or '未设置'}")
    print(f"DashScope OCR MODEL NAME: {DASHSCOPE_OCR_MODEL_NAME or '未设置'}")
    print(f"DashScope API 密钥: {'已加载' if DASHSCOPE_API_KEY else '未找到'}")
    print(f"DashScope API URL: {DASHSCOPE_API_URL or '未设置'}")
    print(f"DeepSeek MODEL NAME: {DEEPSEEK_MODEL_NAME or '未设置'}")
    print(f"DeepSeek API 密钥: {'已加载' if DEEPSEEK_API_KEY else '未找到'}")
    print(f"DeepSeek API URL: {DEEPSEEK_API_URL or '未设置'}")

    print("\n--- OCR 配置加载状态 ---")
    print(f"默认OCR模型: {DEFAULT_OCR_MODEL}")
    print(f"OCR最大重试次数: {OCR_MAX_RETRIES}")
    print(f"OCR超时时间: {OCR_TIMEOUT}秒")
    print(f"阿里云OCR API密钥: {'已加载' if ALIYUN_OCR_API_KEY else '未找到'}")
    print(f"百度OCR API密钥: {'已加载' if BAIDU_OCR_API_KEY else '未找到'}")
    print(f"Google Vision API密钥: {'已加载' if GOOGLE_VISION_API_KEY else '未找到'}")
    print(f"Azure Vision API密钥: {'已加载' if AZURE_VISION_API_KEY else '未找到'}")

    # 4. 再次调用，应该不会重复加载
    print("\n--- 尝试重复加载 ---")
    load_config()

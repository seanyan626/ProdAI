# tests/models/llm/test_dashscope_llm.py
# DashScopeLLM 类的单元测试 (基于 Langchain 封装)
import logging
from unittest.mock import MagicMock, patch

import pytest

try:
    from configs.config import load_config, DASHSCOPE_API_KEY
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

from core.models.llm.dashscope_llm import DashScopeLLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


# --- 模拟 ChatOpenAI ---
@pytest.fixture
def mock_chat_openai_for_dashscope():
    """创建一个 ChatOpenAI 的 MagicMock 实例用于 DashScope 测试。"""
    mock_client = MagicMock(spec=ChatOpenAI)
    mock_bound_client = MagicMock(spec=ChatOpenAI)

    mock_bound_client.invoke.return_value = AIMessage(content="模拟的DashScope响应内容")
    mock_client.bind.return_value = mock_bound_client
    mock_client.invoke.return_value = AIMessage(content="模拟的DashScope响应内容(无绑定)")

    with patch('langchain_openai.ChatOpenAI', return_value=mock_client) as patched_constructor:
        yield patched_constructor, mock_client


# --- DashScopeLLM 初始化测试 ---
def test_dashscopellm_initialization_success(mock_chat_openai_for_dashscope):
    logger.info("测试 DashScopeLLM (Langchain版) 成功初始化...")
    patched_constructor, _ = mock_chat_openai_for_dashscope

    llm = DashScopeLLM(
        model_name="qwen-test",
        api_key="test_ds_key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1
    )
    assert llm.model_name == "qwen-test"

    patched_constructor.assert_called_once()
    call_args, call_kwargs = patched_constructor.call_args
    assert call_kwargs.get("model_name") == "qwen-test"
    assert call_kwargs.get("openai_api_key") == "test_ds_key"
    assert call_kwargs.get("openai_api_base") == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert call_kwargs.get("temperature") == 0.1

    logger.info("DashScopeLLM (Langchain版) 初始化成功测试通过。")


def test_dashscopellm_initialization_no_apikey_raises_valueerror():
    logger.info("测试 DashScopeLLM (Langchain版) 初始化时缺少 API 密钥...")
    with patch('configs.config.DASHSCOPE_API_KEY', None):
        with pytest.raises(ValueError, match="DashScope API 密钥缺失"):
            DashScopeLLM(api_key=None)
    logger.info("缺少 API 密钥时正确引发 ValueError。")


# --- DashScopeLLM 方法测试 ---
@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_dashscopellm_chat_method(mock_chat_openai_for_dashscope):
    logger.info("测试 DashScopeLLM (Langchain版) chat 方法...")
    _, mock_client = mock_chat_openai_for_dashscope
    llm = DashScopeLLM(api_key="fake_ds_key")

    messages = [{"role": "user", "content": "你好"}]
    response = llm.chat(messages, max_tokens=100)

    mock_client.bind.assert_called_with(max_tokens=100)
    bound_client_mock = mock_client.bind.return_value
    bound_client_mock.invoke.assert_called_once()

    assert response["content"] == "模拟的DashScope响应内容"
    logger.info("DashScopeLLM (Langchain版) chat 方法测试通过。")


@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_dashscopellm_generate_method(mock_chat_openai_for_dashscope):
    logger.info("测试 DashScopeLLM (Langchain版) generate 方法...")
    _, mock_client = mock_chat_openai_for_dashscope
    llm = DashScopeLLM(api_key="fake_ds_key")

    prompt = "这是一个单轮提示。"
    response_content = llm.generate(prompt)

    # generate 内部调用 chat, chat内部调用 _get_configured_client, 它会调用 bind
    # 因为没有运行时参数，bind可能不会被调用，而是直接调用原始client的invoke
    mock_client.invoke.assert_called_once()
    assert response_content == "模拟的DashScope响应内容(无绑定)"
    logger.info("DashScopeLLM (Langchain版) generate 方法测试通过。")


logger.info("DashScopeLLM 的单元测试文件已更新为 Langchain 版本。")

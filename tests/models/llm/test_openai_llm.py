# tests/models/llm/test_openai_llm.py
# OpenAILLM 类的单元测试
import logging
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

try:
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

from core.models.llm.openai_llm import OpenAILLM, _convert_dict_messages_to_langchain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# --- 辅助函数测试 (已在openai_llm.py中，但在这里保留一个副本或引用也可以) ---
def test_convert_dict_messages_to_langchain():
    logger.info("测试 _convert_dict_messages_to_langchain 辅助函数...")
    dict_messages = [
        {"role": "system", "content": "系统消息"},
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "AI消息"},
    ]
    lc_messages = _convert_dict_messages_to_langchain(dict_messages)
    assert len(lc_messages) == 3
    assert isinstance(lc_messages[0], SystemMessage)
    assert isinstance(lc_messages[1], HumanMessage)
    assert isinstance(lc_messages[2], AIMessage)
    logger.info("_convert_dict_messages_to_langchain 测试通过。")


# --- OpenAILLM 测试固件 ---
@pytest.fixture
def mock_chat_openai_instance():
    mock_client = MagicMock(spec=ChatOpenAI)
    mock_bound_client = MagicMock(spec=ChatOpenAI)  # 用于模拟 .bind() 后的对象

    mock_bound_client.invoke.return_value = AIMessage(content="模拟的同步响应内容")
    mock_bound_client.ainvoke = AsyncMock(return_value=AIMessage(content="模拟的异步响应内容"))

    # 让原始 client 的 bind 方法返回这个 mock_bound_client
    mock_client.bind.return_value = mock_bound_client
    # 同时，如果没有任何参数绑定，invoke/ainvoke 仍然需要工作
    mock_client.invoke.return_value = AIMessage(content="模拟的同步响应内容(无绑定)")
    mock_client.ainvoke = AsyncMock(return_value=AIMessage(content="模拟的异步响应内容(无绑定)"))

    return mock_client


@pytest.fixture
def patched_chat_openai(mock_chat_openai_instance):
    with patch('langchain_openai.ChatOpenAI', return_value=mock_chat_openai_instance) as patched_constructor:
        yield patched_constructor, mock_chat_openai_instance


# --- OpenAILLM 初始化测试 ---
def test_openaillm_initialization_success(patched_chat_openai):
    logger.info("测试 OpenAILLM 成功初始化...")
    patched_constructor, _ = patched_chat_openai
    llm = OpenAILLM(model_name="gpt-test", api_key="test_key", temperature=0.1, max_tokens=50)
    patched_constructor.assert_called_once_with(
        model_name="gpt-test",
        openai_api_key="test_key",
        temperature=0.1,
        max_tokens=50
    )
    logger.info("OpenAILLM 初始化成功。")


def test_openaillm_initialization_no_apikey_raises_valueerror():
    logger.info("测试 OpenAILLM 初始化时缺少 API 密钥...")
    with patch('configs.config.OPENAI_API_KEY', None):
        with pytest.raises(ValueError, match="OpenAI API 密钥缺失"):
            OpenAILLM(api_key=None)
    logger.info("缺少 API 密钥时按预期引发 ValueError。")


# --- OpenAILLM 方法测试 ---
@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_openaillm_generate_method(patched_chat_openai):
    logger.info("测试 OpenAILLM generate 方法...")
    _, mock_client = patched_chat_openai
    llm = OpenAILLM(api_key="fake_key")

    prompt = "你好！"
    response = llm.generate(prompt, max_tokens=10, stop_sequences=["stop"])

    # 检查是否调用了 bind (因为传递了运行时参数)
    mock_client.bind.assert_called_with(max_tokens=10, stop=["stop"])
    # 获取 bind 返回的模拟对象，并检查它的 invoke 是否被调用
    bound_client_mock = mock_client.bind.return_value
    bound_client_mock.invoke.assert_called_once()
    args, _ = bound_client_mock.invoke.call_args
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == prompt
    assert response == "模拟的同步响应内容"  # 这是 mock_bound_client.invoke 的返回值
    logger.info("OpenAILLM generate 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
@pytest.mark.asyncio
async def test_openaillm_agenerate_method(patched_chat_openai):
    logger.info("测试 OpenAILLM agenerate 方法...")
    _, mock_client = patched_chat_openai
    llm = OpenAILLM(api_key="fake_key")

    prompt = "异步你好！"
    response = await llm.agenerate(prompt, temperature=0.9)

    mock_client.bind.assert_called_with(temperature=0.9)
    bound_client_mock = mock_client.bind.return_value
    bound_client_mock.ainvoke.assert_called_once()
    args, _ = bound_client_mock.ainvoke.call_args
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == prompt
    assert response == "模拟的异步响应内容"
    logger.info("OpenAILLM agenerate 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_openaillm_chat_method(patched_chat_openai):
    logger.info("测试 OpenAILLM chat 方法...")
    _, mock_client = patched_chat_openai
    # 为绑定的客户端配置特定的 invoke 返回值
    bound_client_mock = MagicMock()
    bound_client_mock.invoke.return_value = AIMessage(
        content="模拟的聊天回复", response_metadata={"token_usage": {"total_tokens": 20}}
    )
    mock_client.bind.return_value = bound_client_mock

    llm = OpenAILLM(api_key="fake_key")
    messages = [{"role": "user", "content": "聊天吧！"}]
    response = llm.chat(messages, max_tokens=100)

    mock_client.bind.assert_called_with(max_tokens=100)
    bound_client_mock.invoke.assert_called_once()
    args, _ = bound_client_mock.invoke.call_args
    assert isinstance(args[0][0], HumanMessage)

    assert response["role"] == "assistant"
    assert response["content"] == "模拟的聊天回复"
    assert response["metadata"]["token_usage"]["total_tokens"] == 20
    logger.info("OpenAILLM chat 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
@pytest.mark.asyncio
async def test_openaillm_achat_method(patched_chat_openai):
    logger.info("测试 OpenAILLM achat 方法...")
    _, mock_client = patched_chat_openai
    bound_client_mock = MagicMock()
    bound_client_mock.ainvoke = AsyncMock(return_value=AIMessage(
        content="模拟的异步聊天回复", response_metadata={"finish_reason": "length"}
    ))
    mock_client.bind.return_value = bound_client_mock

    llm = OpenAILLM(api_key="fake_key")
    messages = [{"role": "system", "content": "系统提示"}, {"role": "user", "content": "异步聊天"}]
    response = await llm.achat(messages)

    mock_client.bind.assert_called()  # 至少被调用一次，参数取决于是否有运行时覆盖
    bound_client_mock.ainvoke.assert_called_once()
    args, _ = bound_client_mock.ainvoke.call_args
    assert isinstance(args[0][0], SystemMessage)

    assert response["role"] == "assistant"
    assert response["content"] == "模拟的异步聊天回复"
    assert response["metadata"]["finish_reason"] == "length"
    logger.info("OpenAILLM achat 方法测试通过。")


logger.info("OpenAILLM 的单元测试已更新以反映对 .bind() 的使用。")

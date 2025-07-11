# tests/test_llms.py
# 针对 src/llms/openai_llm.py 中 OpenAILLM 类的单元测试
import pytest
import logging
from unittest.mock import MagicMock, patch, AsyncMock

# 确保配置和日志已设置 (通常在 conftest.py 中完成)
try:
    from configs.config import load_config, OPENAI_API_KEY
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

# 被测试的类
from core.models.language.openai_language_model import OpenAILanguageModel, _convert_dict_messages_to_langchain # 更新路径和类名
# Langchain 核心类，用于类型提示和模拟
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# --- 辅助函数测试 ---
def test_convert_dict_messages_to_langchain(): # 这个辅助函数现在在 openai_language_model.py 内部，所以这个测试仍然有效
    logger.info("测试 _convert_dict_messages_to_langchain 辅助函数...")
    dict_messages = [
        {"role": "system", "content": "系统消息"},
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "AI消息"},
        {"role": "unknown_role", "content": "未知角色消息"}, # 测试未知角色
        {"content": "无角色消息"} # 测试无角色（应默认为user）
    ]
    lc_messages = _convert_dict_messages_to_langchain(dict_messages)
    assert len(lc_messages) == 5
    assert isinstance(lc_messages[0], SystemMessage)
    assert lc_messages[0].content == "系统消息"
    assert isinstance(lc_messages[1], HumanMessage)
    assert lc_messages[1].content == "用户消息"
    assert isinstance(lc_messages[2], AIMessage)
    assert lc_messages[2].content == "AI消息"
    assert isinstance(lc_messages[3], HumanMessage) # 未知角色应转为 HumanMessage
    assert lc_messages[3].content == "未知角色消息"
    assert isinstance(lc_messages[4], HumanMessage) # 无角色应转为 HumanMessage
    assert lc_messages[4].content == "无角色消息"
    logger.info("_convert_dict_messages_to_langchain 测试通过。")

# --- OpenAILLM 测试固件 ---
@pytest.fixture
def mock_chat_openai_instance():
    """创建一个 ChatOpenAI 的 MagicMock 实例。"""
    mock_client = MagicMock(spec=ChatOpenAI)
    # 为 invoke 和 ainvoke 设置默认返回值
    mock_client.invoke.return_value = AIMessage(content="模拟的同步响应内容")
    # ainvoke 需要一个 AsyncMock 作为返回值，或者直接返回一个 awaitable
    async_response_message = AIMessage(content="模拟的异步响应内容")
    mock_client.ainvoke = AsyncMock(return_value=async_response_message)
    return mock_client

@pytest.fixture
def patched_chat_openai(mock_chat_openai_instance):
    """Patch ChatOpenAI 的构造函数以返回 mock_chat_openai_instance。"""
    with patch('langchain_openai.ChatOpenAI', return_value=mock_chat_openai_instance) as patched_constructor:
        yield patched_constructor, mock_chat_openai_instance

# --- OpenAILanguageModel 初始化测试 ---
def test_openailanguagemodel_initialization_success(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel 成功初始化...") # 类名更新
    patched_constructor, _ = patched_chat_openai

    llm = OpenAILanguageModel( # 类名更新
        model_name="gpt-test",
        api_key="test_api_key",
        temperature=0.5,
        max_tokens=100,
        llm_specific_kwargs={"top_p": 0.9}
    )
    assert llm.model_name == "gpt-test"
    assert llm.api_key == "test_api_key"
    # temperature 和 max_tokens 现在存储在 config 中
    assert llm.config.get("temperature") == 0.5
    assert llm.config.get("max_tokens") == 100
    assert llm.config["llm_specific_kwargs"] == {"top_p": 0.9}

    patched_constructor.assert_called_once_with(
        model_name="gpt-test",
        openai_api_key="test_api_key",
        temperature=0.5,
        max_tokens=100,
        top_p=0.9
    )
    logger.info("OpenAILanguageModel 初始化成功测试通过。") # 类名更新

def test_openailanguagemodel_initialization_no_apikey_raises_valueerror(): # 函数名更新
    logger.info("测试 OpenAILanguageModel 初始化时缺少 API 密钥是否引发 ValueError...") # 类名更新
    with patch('configs.config.OPENAI_API_KEY', None):
         with pytest.raises(ValueError, match="OpenAI API 密钥缺失"):
            OpenAILanguageModel(api_key=None) # 类名更新
    logger.info("缺少 API 密钥时正确引发 ValueError。")


# --- OpenAILanguageModel 方法测试 ---
@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE", reason="需要有效的 OpenAI API 密钥进行部分模拟测试")
def test_openailanguagemodel_generate_method(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel generate 方法...") # 类名更新
    _, mock_client = patched_chat_openai
    llm = OpenAILanguageModel(api_key="fake_key") # 类名更新

    prompt = "你好，世界！"
    response = llm.generate(prompt, max_tokens=50, stop_sequences=["\n"])

    # 验证 mock_client.invoke 是否被正确调用
    # _convert_dict_messages_to_langchain 会将 prompt 转为 [HumanMessage(content=prompt)]
    mock_client.invoke.assert_called_once()
    args, kwargs = mock_client.invoke.call_args
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == prompt
    assert kwargs.get("max_tokens") == 50
    assert kwargs.get("stop") == ["\n"]

    assert response == "模拟的同步响应内容"
    logger.info("OpenAILanguageModel generate 方法测试通过。") # 类名更新

@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE", reason="需要有效的 OpenAI API 密钥")
@pytest.mark.asyncio
async def test_openailanguagemodel_agenerate_method(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel agenerate 方法...") # 类名更新
    _, mock_client = patched_chat_openai
    llm = OpenAILanguageModel(api_key="fake_key") # 类名更新

    prompt = "异步你好！"
    response = await llm.agenerate(prompt, max_tokens=60)

    mock_client.ainvoke.assert_called_once()
    args, kwargs = mock_client.ainvoke.call_args
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == prompt
    assert kwargs.get("max_tokens") == 60

    assert response == "模拟的异步响应内容"
    logger.info("OpenAILanguageModel agenerate 方法测试通过。") # 类名更新

@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE", reason="需要有效的 OpenAI API 密钥")
def test_openailanguagemodel_chat_method(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel chat 方法...") # 类名更新
    _, mock_client = patched_chat_openai
    mock_client.invoke.return_value = AIMessage(
        content="模拟的聊天回复",
        response_metadata={"token_usage": {"total_tokens": 10}}
    )
    llm = OpenAILanguageModel(api_key="fake_key") # 类名更新

    messages = [{"role": "user", "content": "打个招呼吧！"}]
    response = llm.chat(messages, temperature=0.2)

    mock_client.invoke.assert_called_once()
    args, kwargs = mock_client.invoke.call_args
    assert isinstance(args[0][0], HumanMessage)
    assert args[0][0].content == "打个招呼吧！"

    assert response["role"] == "assistant"
    assert response["content"] == "模拟的聊天回复"
    assert response["metadata"]["token_usage"]["total_tokens"] == 10
    assert response["token_usage"]["total_tokens"] == 10
    logger.info("OpenAILanguageModel chat 方法测试通过。") # 类名更新

@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE", reason="需要有效的 OpenAI API 密钥")
@pytest.mark.asyncio
async def test_openailanguagemodel_achat_method(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel achat 方法...") # 类名更新
    _, mock_client = patched_chat_openai
    mock_client.ainvoke.return_value = AIMessage(
        content="模拟的异步聊天回复",
        response_metadata={"finish_reason": "stop"}
    )
    llm = OpenAILanguageModel(api_key="fake_key") # 类名更新

    messages = [{"role": "system", "content": "你是机器人。"}, {"role": "user", "content": "你是谁？"}]
    response = await llm.achat(messages)

    mock_client.ainvoke.assert_called_once()
    args, kwargs = mock_client.ainvoke.call_args
    assert isinstance(args[0][0], SystemMessage)
    assert isinstance(args[0][1], HumanMessage)

    assert response["role"] == "assistant"
    assert response["content"] == "模拟的异步聊天回复"
    assert response["metadata"]["finish_reason"] == "stop"
    logger.info("OpenAILanguageModel achat 方法测试通过。") # 类名更新

@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE", reason="需要有效的 OpenAI API 密钥")
def test_openailanguagemodel_generate_handles_api_error(patched_chat_openai): # 函数名更新
    logger.info("测试 OpenAILanguageModel generate 方法处理 API 错误...") # 类名更新
    _, mock_client = patched_chat_openai
    mock_client.invoke.side_effect = Exception("模拟API连接错误")
    llm = OpenAILanguageModel(api_key="fake_key") # 类名更新

    response = llm.generate("一个会出错的提示")
    assert "错误：LLM 生成失败" in response
    assert "模拟API连接错误" in response
    logger.info("OpenAILanguageModel generate 方法 API 错误处理测试通过。") # 类名更新

logger.info("LLM 单元测试文件 (tests/test_llms.py) 框架完成。")

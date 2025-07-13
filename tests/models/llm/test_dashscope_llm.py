# tests/models/llm/test_dashscope_llm.py
# DashScopeLLM 类的单元测试
import pytest
import logging
from unittest.mock import MagicMock, patch
from http import HTTPStatus

# 确保配置和日志已设置
try:
    from configs.config import load_config, DASHSCOPE_API_KEY
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

from core.models.llm.dashscope_llm import DashScopeLLM

logger = logging.getLogger(__name__)

# --- 模拟 DashScope SDK ---
@pytest.fixture
def mock_dashscope_generation():
    """模拟 dashscope.Generation.call 的测试固件。"""
    # 创建一个模拟的响应对象，模仿 dashscope.api_entities.dashscope_response.GenerationResponse
    mock_response = MagicMock()
    mock_response.status_code = HTTPStatus.OK
    # 构建嵌套的模拟对象以匹配 response.output.choices[0].message.content
    mock_message = MagicMock()
    mock_message.content = "模拟的DashScope响应内容"
    mock_message.role = "assistant"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_output = MagicMock()
    mock_output.choices = [mock_choice]
    mock_response.output = mock_output
    # 模拟 usage
    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 20
    mock_response.usage = mock_usage

    with patch('dashscope.Generation.call', return_value=mock_response) as mock_call:
        yield mock_call, mock_response

# --- DashScopeLLM 初始化测试 ---
def test_dashscopellm_initialization_success():
    logger.info("测试 DashScopeLLM 成功初始化...")
    with patch('dashscope.api_key', new_callable=MagicMock) as mock_api_key_setter:
        llm = DashScopeLLM(model_name="qwen-test", api_key="test_ds_key")
        assert llm.model_name == "qwen-test"
        assert llm.api_key == "test_ds_key"
        # 验证 dashscope.api_key 是否被正确设置
        assert dashscope.api_key == "test_ds_key"
    logger.info("DashScopeLLM 初始化成功测试通过。")

def test_dashscopellm_initialization_no_apikey_raises_valueerror():
    logger.info("测试 DashScopeLLM 初始化时缺少 API 密钥是否引发 ValueError...")
    with patch('configs.config.DASHSCOPE_API_KEY', None):
         with pytest.raises(ValueError, match="DashScope API 密钥缺失"):
            DashScopeLLM(api_key=None)
    logger.info("缺少 API 密钥时正确引发 ValueError。")

# --- DashScopeLLM 方法测试 ---
@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY", reason="部分模拟测试可能仍需有效配置结构")
def test_dashscopellm_chat_method(mock_dashscope_generation):
    logger.info("测试 DashScopeLLM chat 方法...")
    mock_call, _ = mock_dashscope_generation
    llm = DashScopeLLM(api_key="fake_ds_key")

    messages = [{"role": "user", "content": "你好"}]
    response = llm.chat(messages, temperature=0.5, max_tokens=100)

    # 验证 dashscope.Generation.call 是否以期望的参数被调用
    mock_call.assert_called_once()
    call_args, call_kwargs = mock_call.call_args
    assert call_kwargs.get("model") == "qwen-turbo" # 默认模型
    assert call_kwargs.get("messages") == messages
    assert call_kwargs.get("result_format") == "message"
    assert call_kwargs.get("temperature") == 0.5
    assert call_kwargs.get("max_tokens") == 100

    # 验证返回格式是否正确
    assert response["role"] == "assistant"
    assert response["content"] == "模拟的DashScope响应内容"
    assert response["token_usage"]["total_tokens"] == 30
    logger.info("DashScopeLLM chat 方法测试通过。")

@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY", reason="部分模拟测试可能仍需有效配置结构")
def test_dashscopellm_generate_method(mock_dashscope_generation):
    logger.info("测试 DashScopeLLM generate 方法...")
    mock_call, _ = mock_dashscope_generation
    llm = DashScopeLLM(api_key="fake_ds_key")

    prompt = "这是一个单轮提示。"
    response_content = llm.generate(prompt)

    # generate 内部调用 chat，所以我们验证对 chat 的模拟调用
    mock_call.assert_called_once()
    call_args, call_kwargs = mock_call.call_args
    expected_messages = [{"role": "user", "content": prompt}]
    assert call_kwargs.get("messages") == expected_messages

    assert response_content == "模拟的DashScope响应内容"
    logger.info("DashScopeLLM generate 方法测试通过。")

@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY", reason="部分模拟测试可能仍需有效配置结构")
@pytest.mark.asyncio
async def test_dashscopellm_achat_method(mock_dashscope_generation):
    logger.info("测试 DashScopeLLM achat 方法...")
    # 由于achat是包装的同步调用，我们实际上还是在测试同步调用的逻辑
    mock_call, _ = mock_dashscope_generation
    llm = DashScopeLLM(api_key="fake_ds_key")

    messages = [{"role": "user", "content": "异步你好"}]
    response = await llm.achat(messages, temperature=0.8)

    # 验证同步的 call 方法被调用
    mock_call.assert_called_once()
    call_args, call_kwargs = mock_call.call_args
    assert call_kwargs.get("temperature") == 0.8

    assert response["content"] == "模拟的DashScope响应内容"
    logger.info("DashScopeLLM achat 方法测试通过。")

@pytest.mark.skipif(not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY", reason="部分模拟测试可能仍需有效配置结构")
def test_dashscopellm_chat_handles_api_error(mock_dashscope_generation):
    logger.info("测试 DashScopeLLM chat 方法处理 API 错误...")
    mock_call, mock_response = mock_dashscope_generation

    # 修改模拟响应以表示错误
    mock_response.status_code = HTTPStatus.BAD_REQUEST
    mock_response.code = "InvalidParameter"
    mock_response.message = "模拟的无效参数错误"

    llm = DashScopeLLM(api_key="fake_ds_key")
    response = llm.chat([{"role": "user", "content": "一个会出错的提示"}])

    assert "error" in response
    assert response["content"] == f"错误: API 调用失败 - {mock_response.message}"
    logger.info("DashScopeLLM chat 方法 API 错误处理测试通过。")

logger.info("DashScopeLLM 的单元测试文件创建完毕。")

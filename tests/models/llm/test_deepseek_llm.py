# tests/models/llm/test_deepseek_llm.py
# DeepSeekLLM 类的单元测试
import pytest
import logging
from unittest.mock import MagicMock, patch

try:
    from configs.config import load_config, DEEPSEEK_API_KEY
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

from core.models.llm.deepseek_llm import DeepSeekLLM
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# --- 模拟 ChatOpenAI ---
@pytest.fixture
def mock_chat_openai_for_deepseek():
    """创建一个 ChatOpenAI 的 MagicMock 实例用于 DeepSeek 测试。"""
    mock_client = MagicMock(spec=ChatOpenAI)
    # 模拟 .bind().invoke() 链
    mock_bound_client = MagicMock()
    mock_bound_client.invoke.return_value.content = "模拟的DeepSeek响应"
    mock_client.bind.return_value = mock_bound_client
    mock_client.invoke.return_value.content = "模拟的DeepSeek响应(无绑定)"

    with patch('langchain_openai.ChatOpenAI', return_value=mock_client) as patched_constructor:
        yield patched_constructor, mock_client

# --- DeepSeekLLM 初始化测试 ---
def test_deepseekllm_initialization_success(mock_chat_openai_for_deepseek):
    logger.info("测试 DeepSeekLLM 成功初始化...")
    patched_constructor, _ = mock_chat_openai_for_deepseek

    llm = DeepSeekLLM(
        model_name="deepseek-coder",
        api_key="test_ds_key",
        base_url="https://api.deepseek.com/v1/test",
        temperature=0.1
    )
    assert llm.model_name == "deepseek-coder"
    assert llm.api_key == "test_ds_key"
    assert llm.config.get("base_url") == "https://api.deepseek.com/v1/test"

    # 验证 ChatOpenAI 构造函数是否以期望的参数被调用
    patched_constructor.assert_called_once()
    call_args, call_kwargs = patched_constructor.call_args
    assert call_kwargs.get("model_name") == "deepseek-coder"
    assert call_kwargs.get("openai_api_key") == "test_ds_key"
    assert call_kwargs.get("openai_api_base") == "https://api.deepseek.com/v1/test"
    assert call_kwargs.get("temperature") == 0.1

    logger.info("DeepSeekLLM 初始化成功测试通过。")

def test_deepseekllm_initialization_no_apikey_raises_valueerror():
    logger.info("测试 DeepSeekLLM 初始化时缺少 API 密钥是否引发 ValueError...")
    with patch('configs.config.DEEPSEEK_API_KEY', None):
         with pytest.raises(ValueError, match="DeepSeek API 密钥缺失"):
            DeepSeekLLM(api_key=None)
    logger.info("缺少 API 密钥时正确引发 ValueError。")

# --- DeepSeekLLM 方法测试 ---
@pytest.mark.skipif(not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE", reason="部分模拟测试可能仍需有效配置结构")
def test_deepseekllm_chat_method(mock_chat_openai_for_deepseek):
    logger.info("测试 DeepSeekLLM chat 方法...")
    _, mock_client = mock_chat_openai_for_deepseek
    llm = DeepSeekLLM(api_key="fake_ds_key")

    messages = [{"role": "user", "content": "你好"}]
    response = llm.chat(messages, max_tokens=50)

    # 验证 .bind(max_tokens=50) 被调用
    mock_client.bind.assert_called_with(max_tokens=50)
    # 验证 .bind() 返回的对象的 .invoke() 被调用
    bound_client_mock = mock_client.bind.return_value
    bound_client_mock.invoke.assert_called_once()

    assert response["content"] == "模拟的DeepSeek响应"
    logger.info("DeepSeekLLM chat 方法测试通过。")

logger.info("DeepSeekLLM 的单元测试文件创建完毕。")

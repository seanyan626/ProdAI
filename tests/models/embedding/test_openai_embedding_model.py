# tests/models/embedding/test_openai_embedding_model.py
# OpenAIEmbeddingModel 的单元测试
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

from core.models.embedding.openai_embedding_model import OpenAIEmbeddingModel
from langchain_openai import OpenAIEmbeddings  # 用于模拟

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_openai_embeddings_instance():
    """创建一个 OpenAIEmbeddings 的 MagicMock 实例。"""
    mock_client = MagicMock(spec=OpenAIEmbeddings)
    mock_client.embed_documents.return_value = [[0.1, 0.2, 0.3]]  # 示例嵌入
    mock_client.embed_query.return_value = [0.1, 0.2, 0.3]

    async_embed_docs_result = [[0.1, 0.2, 0.3, 0.4]]
    mock_client.aembed_documents = AsyncMock(return_value=async_embed_docs_result)

    async_embed_query_result = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_client.aembed_query = AsyncMock(return_value=async_embed_query_result)
    return mock_client


@pytest.fixture
def patched_openai_embeddings(mock_openai_embeddings_instance):
    """Patch OpenAIEmbeddings 的构造函数以返回 mock_openai_embeddings_instance。"""
    with patch('langchain_openai.OpenAIEmbeddings',
               return_value=mock_openai_embeddings_instance) as patched_constructor:
        yield patched_constructor, mock_openai_embeddings_instance


# --- OpenAIEmbeddingModel 初始化测试 ---
def test_openai_embedding_model_initialization(patched_openai_embeddings):
    logger.info("测试 OpenAIEmbeddingModel 成功初始化...")
    patched_constructor, _ = patched_openai_embeddings

    model = OpenAIEmbeddingModel(
        model_name="text-embedding-test",
        api_key="test_api_key_emb",
        embedding_specific_kwargs={"chunk_size": 500}
    )
    assert model.model_name == "text-embedding-test"
    assert model.api_key == "test_api_key_emb"

    patched_constructor.assert_called_once()
    # 检查构造函数调用时的参数，移除 self.config 中已经被显式使用的参数
    expected_call_kwargs = {
        "model": "text-embedding-test",
        "openai_api_key": "test_api_key_emb",
        "chunk_size": 500
    }
    # 获取实际调用参数
    actual_call_args, actual_call_kwargs = patched_constructor.call_args
    # 比较关键参数（Pydantic模型可能会有默认值，所以只比较我们关心的）
    for key, value in expected_call_kwargs.items():
        assert actual_call_kwargs.get(key) == value

    logger.info("OpenAIEmbeddingModel 初始化成功测试通过。")


def test_openai_embedding_model_initialization_no_apikey_raises_valueerror():
    logger.info("测试 OpenAIEmbeddingModel 初始化时缺少 API 密钥是否引发 ValueError...")
    with patch('configs.config.OPENAI_API_KEY', None):
        with pytest.raises(ValueError, match="OpenAI API 密钥缺失"):
            OpenAIEmbeddingModel(api_key=None)
    logger.info("缺少 API 密钥时正确引发 ValueError。")


# --- OpenAIEmbeddingModel 方法测试 ---
@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_embed_documents(patched_openai_embeddings):
    logger.info("测试 embed_documents 方法...")
    _, mock_client = patched_openai_embeddings
    model = OpenAIEmbeddingModel(api_key="fake_key_emb")

    texts = ["文本1", "文本2"]
    embeddings = model.embed_documents(texts)

    mock_client.embed_documents.assert_called_once_with(texts)
    assert embeddings == [[0.1, 0.2, 0.3]]  # 匹配模拟实例的返回值
    logger.info("embed_documents 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
@pytest.mark.asyncio
async def test_aembed_documents(patched_openai_embeddings):
    logger.info("测试 aembed_documents 方法...")
    _, mock_client = patched_openai_embeddings
    model = OpenAIEmbeddingModel(api_key="fake_key_emb")

    texts = ["异步文本1", "异步文本2"]
    embeddings = await model.aembed_documents(texts)

    mock_client.aembed_documents.assert_called_once_with(texts)
    assert embeddings == [[0.1, 0.2, 0.3, 0.4]]
    logger.info("aembed_documents 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
def test_embed_query(patched_openai_embeddings):
    logger.info("测试 embed_query 方法...")
    _, mock_client = patched_openai_embeddings
    model = OpenAIEmbeddingModel(api_key="fake_key_emb")

    text = "查询文本"
    embedding = model.embed_query(text)

    mock_client.embed_query.assert_called_once_with(text)
    assert embedding == [0.1, 0.2, 0.3]
    logger.info("embed_query 方法测试通过。")


@pytest.mark.skipif(not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE",
                    reason="部分模拟测试可能仍需有效配置结构")
@pytest.mark.asyncio
async def test_aembed_query(patched_openai_embeddings):
    logger.info("测试 aembed_query 方法...")
    _, mock_client = patched_openai_embeddings
    model = OpenAIEmbeddingModel(api_key="fake_key_emb")

    text = "异步查询文本"
    embedding = await model.aembed_query(text)

    mock_client.aembed_query.assert_called_once_with(text)
    assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    logger.info("aembed_query 方法测试通过。")


logger.info("OpenAIEmbeddingModel 的单元测试文件创建完毕。")

# tests/models/ocr/test_openai_ocr_model.py
# OpenAIOCRModel 的单元测试 (当前为骨架)
import logging

import pytest

try:
    from configs.config import load_config
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为测试导入配置/日志模块。")

from core.models.ocr.base_ocr_model import OCRResult
from core.models.ocr.openai_ocr_model import OpenAIOCRModel

logger = logging.getLogger(__name__)


@pytest.fixture
def openai_ocr_model_skeleton():
    """提供 OpenAIOCRModel 的骨架实例。"""
    # 由于是骨架，这里不需要模拟API密钥等
    return OpenAIOCRModel(model_name="test_ocr_skeleton")


def test_openai_ocr_model_initialization(openai_ocr_model_skeleton):
    logger.info("测试 OpenAIOCRModel (骨架) 初始化...")
    model = openai_ocr_model_skeleton
    assert model is not None
    assert model.model_name == "test_ocr_skeleton"
    model_info = model.get_model_info()
    assert model_info["type"] == "OCRModel"
    logger.info("OpenAIOCRModel (骨架) 初始化测试通过。")


@pytest.mark.skip(reason="OpenAIOCRModel recognize_text 方法是骨架，待具体实现后测试。")
def test_openai_ocr_model_recognize_text_skeleton(openai_ocr_model_skeleton):
    logger.info("测试 OpenAIOCRModel (骨架) 的 recognize_text 方法...")
    model = openai_ocr_model_skeleton
    dummy_image_path = "dummy.png"

    # 当前骨架会返回占位符文本
    expected_placeholder_text = "[OpenAI OCR 功能待实现]"

    result = model.recognize_text(dummy_image_path)
    assert isinstance(result, OCRResult)
    assert result.full_text == expected_placeholder_text
    assert isinstance(result.segments, list)

    # 模拟字节输入
    dummy_image_bytes = b"dummy bytes"
    result_bytes = model.recognize_text(dummy_image_bytes)
    assert result_bytes.full_text == expected_placeholder_text
    logger.info("OpenAIOCRModel (骨架) recognize_text 方法基本调用测试通过 (返回占位符)。")


@pytest.mark.skip(reason="OpenAIOCRModel arecognize_text 方法是骨架，待具体实现后测试。")
@pytest.mark.asyncio
async def test_openai_ocr_model_arecognize_text_skeleton(openai_ocr_model_skeleton):
    logger.info("测试 OpenAIOCRModel (骨架) 的 arecognize_text 方法...")
    model = openai_ocr_model_skeleton
    dummy_image_path = "dummy.png"

    expected_placeholder_text = "[OpenAI OCR 异步功能待实现]"

    result = await model.arecognize_text(dummy_image_path)
    assert isinstance(result, OCRResult)
    assert result.full_text == expected_placeholder_text

    dummy_image_bytes = b"dummy bytes"
    result_bytes = await model.arecognize_text(dummy_image_bytes)
    assert result_bytes.full_text == expected_placeholder_text
    logger.info("OpenAIOCRModel (骨架) arecognize_text 方法基本调用测试通过 (返回占位符)。")


logger.info("OpenAIOCRModel (骨架) 的测试文件创建完毕。")

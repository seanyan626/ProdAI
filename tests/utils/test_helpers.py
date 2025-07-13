# tests/utils/test_helpers.py
# 辅助函数模块的测试
import json
import logging
import os

import pytest

# 确保如果任何底层模块需要配置，则加载配置，并设置日志记录。
try:
    from configs.config import load_config
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为辅助函数测试导入配置/日志模块。这可能会影响模块行为。")

from core.utils.helpers import load_json_file, save_json_file, truncate_text  # src -> core

logger = logging.getLogger(__name__)


# --- 测试固件 ---

@pytest.fixture
def temp_json_file(tmp_path):
    """为测试创建一个临时的 JSON 文件路径。"""
    return tmp_path / "test_data_测试数据.json"


# --- JSON 辅助函数测试 ---
# 注意：由于 src/utils/helpers.py 中的实现已移除，以下测试将被跳过。
# 在 helpers.py 中实现相应功能后，请移除 @pytest.mark.skip 标记。

@pytest.mark.skip(reason="src/utils/helpers.py 中的 JSON 函数实现已移除。")
def test_save_and_load_json_file(temp_json_file):
    logger.info(f"正在使用以下文件测试 save_json_file 和 load_json_file: {temp_json_file}")
    test_data = {"名称": "AI 项目", "版本": "1.0", "模块": ["agents", "llms", "rag"], "语言": "中文"}

    save_json_file(test_data, str(temp_json_file))
    assert os.path.exists(temp_json_file)
    logger.info("JSON 数据成功保存。")

    loaded_data = load_json_file(str(temp_json_file))
    assert loaded_data == test_data
    logger.info("JSON 数据成功加载并与原始数据匹配。")


@pytest.mark.skip(reason="src/utils/helpers.py 中的 JSON 函数实现已移除。")
def test_load_json_file_not_found(tmp_path):
    logger.info("正在使用不存在的文件测试 load_json_file。")
    non_existent_file = tmp_path / "does_not_exist_不存在的文件.json"
    with pytest.raises(FileNotFoundError):  # 实际错误可能因pass占位符而不同
        load_json_file(str(non_existent_file))
    logger.info("按预期引发了 FileNotFoundError (或类似错误，因实现为空)。")


@pytest.mark.skip(reason="src/utils/helpers.py 中的 JSON 函数实现已移除。")
def test_load_json_file_invalid_json(tmp_path):
    logger.info("正在使用无效的 JSON 文件测试 load_json_file。")
    invalid_json_file = tmp_path / "invalid_无效文件.json"
    with open(invalid_json_file, "w", encoding="utf-8") as f:
        f.write("{'名称': '测试', '值': 123,}")

    with pytest.raises(json.JSONDecodeError):  # 实际错误可能因pass占位符而不同
        load_json_file(str(invalid_json_file))
    logger.info("对于无效 JSON 按预期引发了 JSONDecodeError (或类似错误，因实现为空)。")


# --- 文本辅助函数测试 ---
# 注意：由于 src/utils/helpers.py 中的实现已移除，以下测试将被跳过。
# 在 helpers.py 中实现相应功能后，请移除 @pytest.mark.skip 标记。

@pytest.mark.skip(reason="src/utils/helpers.py 中的文本函数实现已移除。")
def test_truncate_text_no_truncation_needed():
    logger.info("正在测试无需截断时的 truncate_text。")
    text = "短文本。"
    max_length = 20
    truncated = truncate_text(text, max_length)
    # assert truncated == text # 实际断言取决于占位符实现
    logger.info(f"结果: '{truncated}' (预期: '{text}')")


@pytest.mark.skip(reason="src/utils/helpers.py 中的文本函数实现已移除。")
def test_truncate_text_truncation_occurs():
    logger.info("正在测试应发生截断时的 truncate_text。")
    text = "这是一个相当长的字符串，绝对需要被截断。"
    max_length = 10
    expected_ellipsis = "..."
    # expected_text = text[:max_length - len(expected_ellipsis)] + expected_ellipsis
    truncated = truncate_text(text, max_length)
    # assert truncated == expected_text # 实际断言取决于占位符实现
    # assert len(truncated) == max_length
    logger.info(f"结果: '{truncated}' (预期包含截断和省略号)")


@pytest.mark.skip(reason="src/utils/helpers.py 中的文本函数实现已移除。")
def test_truncate_text_custom_ellipsis():
    logger.info("正在测试使用自定义省略号的 truncate_text。")
    text = "另一个用于自定义省略号的长示例。"
    max_length = 15
    custom_ellipsis = " (...) "
    # expected_text = text[:max_length - len(custom_ellipsis)] + custom_ellipsis
    truncated = truncate_text(text, max_length, ellipsis=custom_ellipsis)
    # assert truncated == expected_text # 实际断言取决于占位符实现
    # assert len(truncated) == max_length
    logger.info(f"结果: '{truncated}' (预期包含截断和自定义省略号)")


@pytest.mark.skip(reason="src/utils/helpers.py 中的文本函数实现已移除。")
def test_truncate_text_max_length_too_small_for_ellipsis():
    logger.info("正在测试 max_length 小于省略号长度时的 truncate_text。")
    text = "一些文本"
    max_length = 2
    truncated = truncate_text(text, max_length)
    # assert len(truncated) <= max_length # 实际断言取决于占位符实现
    logger.info(f"使用 max_length={max_length}, 省略号='...' 进行截断: '{truncated}'")


@pytest.mark.skip(reason="src/utils/helpers.py 中的文本函数实现已移除。")
def test_truncate_text_empty_string():
    logger.info("正在测试使用空字符串的 truncate_text。")
    text = ""
    truncated = truncate_text(text, 10)
    # assert truncated == "" # 实际断言取决于占位符实现
    logger.info(f"结果: '{truncated}' (预期为空字符串)")


logger.info("辅助函数测试文件已调整为适应骨架代码。多数原测试已跳过。")

# tests/utils/test_helpers.py
# 辅助函数模块的测试
import pytest
import json
import os
import logging

# 确保如果任何底层模块需要配置，则加载配置，并设置日志记录。
try:
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()
except ImportError:
    print("警告: 无法为辅助函数测试导入配置/日志模块。这可能会影响模块行为。")


from src.utils.helpers import load_json_file, save_json_file, truncate_text

logger = logging.getLogger(__name__)

# --- 测试固件 ---

@pytest.fixture
def temp_json_file(tmp_path):
    """为测试创建一个临时的 JSON 文件路径。"""
    return tmp_path / "test_data_测试数据.json"

# --- JSON 辅助函数测试 ---

def test_save_and_load_json_file(temp_json_file):
    logger.info(f"正在使用以下文件测试 save_json_file 和 load_json_file: {temp_json_file}")
    test_data = {"名称": "AI 项目", "版本": "1.0", "模块": ["agents", "llms", "rag"], "语言": "中文"}

    # 测试保存
    try:
        save_json_file(test_data, str(temp_json_file))
        assert os.path.exists(temp_json_file)
        logger.info("JSON 数据成功保存。")
    except Exception as e:
        pytest.fail(f"save_json_file 引发了异常: {e}")

    # 测试加载
    try:
        loaded_data = load_json_file(str(temp_json_file))
        assert loaded_data == test_data
        logger.info("JSON 数据成功加载并与原始数据匹配。")
    except Exception as e:
        pytest.fail(f"load_json_file 引发了异常: {e}")

def test_load_json_file_not_found(tmp_path):
    logger.info("正在使用不存在的文件测试 load_json_file。")
    non_existent_file = tmp_path / "does_not_exist_不存在的文件.json"
    with pytest.raises(FileNotFoundError):
        load_json_file(str(non_existent_file))
    logger.info("按预期引发了 FileNotFoundError。")


def test_load_json_file_invalid_json(tmp_path):
    logger.info("正在使用无效的 JSON 文件测试 load_json_file。")
    invalid_json_file = tmp_path / "invalid_无效文件.json"
    with open(invalid_json_file, "w", encoding="utf-8") as f: # 确保使用utf-8编码写入
        f.write("{'名称': '测试', '值': 123,}") # 无效 JSON (尾随逗号，单引号通常有问题)

    with pytest.raises(json.JSONDecodeError):
        load_json_file(str(invalid_json_file))
    logger.info("对于无效 JSON 按预期引发了 JSONDecodeError。")

# --- 文本辅助函数测试 ---

def test_truncate_text_no_truncation_needed():
    logger.info("正在测试无需截断时的 truncate_text。")
    text = "短文本。"
    max_length = 20
    truncated = truncate_text(text, max_length)
    assert truncated == text
    logger.info(f"结果: '{truncated}' (正确，未截断)")

def test_truncate_text_truncation_occurs():
    logger.info("正在测试应发生截断时的 truncate_text。")
    text = "这是一个相当长的字符串，绝对需要被截断。"
    max_length = 10 # 改小一点以确保截断
    expected_ellipsis = "..."
    # 预期: "这是一个相..." (10 个字符总长, 所以 7 个来自文本 + 3 个省略号)
    expected_text = text[:max_length - len(expected_ellipsis)] + expected_ellipsis

    truncated = truncate_text(text, max_length)
    assert truncated == expected_text
    assert len(truncated) == max_length
    logger.info(f"结果: '{truncated}' (正确，已截断)")

def test_truncate_text_custom_ellipsis():
    logger.info("正在测试使用自定义省略号的 truncate_text。")
    text = "另一个用于自定义省略号的长示例。"
    max_length = 15
    custom_ellipsis = " (...) " # 长度 7
    # 预期: "另一个用 (...)" (15 个字符总长, 所以 8 个来自文本 + 7 个省略号)
    expected_text = text[:max_length - len(custom_ellipsis)] + custom_ellipsis

    truncated = truncate_text(text, max_length, ellipsis=custom_ellipsis)
    assert truncated == expected_text
    assert len(truncated) == max_length
    logger.info(f"结果: '{truncated}' (正确，自定义省略号)")

def test_truncate_text_max_length_too_small_for_ellipsis():
    logger.info("正在测试 max_length 小于省略号长度时的 truncate_text。")
    text = "一些文本"
    max_length = 2 # 小于默认的 "..."
    # 预期行为: 返回截断到 max_length 的文本，不带省略号，或者如果 max_length 允许，则仅返回省略号。
    # helpers.py 中的 truncate_text 已被修正以处理这种情况。

    truncated = truncate_text(text, max_length)
    logger.info(f"使用 max_length={max_length}, 省略号='...' 进行截断: '{truncated}'")
    # 修正后的 truncate_text 应该能正确处理这种情况
    # 如果 max_length=2, ellipsis="...", 应该返回 ".."
    assert len(truncated) <= max_length
    assert truncated == text[:max_length] if max_length < len("...") else ".." # 根据修正后的逻辑调整断言
    logger.info("max_length < len(ellipsis) 时的 truncate_text 行为已审查。")


    # 测试 max_length 正好等于省略号长度的情况
    max_length_equals_ellipsis = 3
    truncated_equal = truncate_text(text, max_length_equals_ellipsis)
    assert truncated_equal == "..."
    assert len(truncated_equal) == max_length_equals_ellipsis
    logger.info(f"使用 max_length={max_length_equals_ellipsis}, 省略号='...' 进行截断: '{truncated_equal}' (正确)")


def test_truncate_text_empty_string():
    logger.info("正在测试使用空字符串的 truncate_text。")
    text = ""
    truncated = truncate_text(text, 10)
    assert truncated == ""
    logger.info(f"结果: '{truncated}' (正确，空字符串)")

logger.info("辅助函数测试完成。")

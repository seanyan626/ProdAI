# src/utils/helpers.py
# 实用工具函数模块
import logging
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    加载 JSON 文件并将其内容作为字典返回。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"成功从 {file_path} 加载 JSON 数据。")
        return data
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"解码 JSON 文件错误: {file_path}")
        raise
    except Exception as e:
        logger.error(f"加载 {file_path} 时发生意外错误: {e}")
        raise

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    将字典保存到 JSON 文件。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False) # ensure_ascii=False 以正确显示中文
        logger.debug(f"成功将 JSON 数据保存到 {file_path}。")
    except Exception as e:
        logger.error(f"保存到 {file_path} 时发生意外错误: {e}")
        raise

# 另一个实用函数示例
def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    将文本截断到最大长度，如果发生截断则添加省略号。
    """
    if len(text) > max_length:
        # 确保省略号本身不会使长度超过 max_length
        if len(ellipsis) >= max_length:
            return ellipsis[:max_length] # 如果省略号比最大长度还长，截断省略号
        return text[:max_length - len(ellipsis)] + ellipsis
    return text

if __name__ == '__main__':
    # 示例用法 (用于直接测试此模块)
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    test_data = {"名称": "测试数据", "版本": 1, "项目": [1, 2, 3], "语言": "中文"}
    test_file_path = "test_helpers_示例.json" # 测试文件名

    logger.info(f"正在保存测试数据到 {test_file_path}...")
    save_json_file(test_data, test_file_path)

    logger.info(f"正在从 {test_file_path} 加载测试数据...")
    loaded_data = load_json_file(test_file_path)
    logger.info(f"加载的数据: {loaded_data}")

    assert test_data == loaded_data
    logger.info("JSON 加载/保存测试成功。")

    original_text = "这是一个非常非常长的字符串，它需要被截断以用于显示目的，否则会超出屏幕范围。"
    truncated = truncate_text(original_text, 30)
    logger.info(f"原文: '{original_text}'")
    logger.info(f"截断后: '{truncated}'")
    assert truncated == "这是一个非常非常长的字符串，它需要被截断以用于显示..." # 根据实际截断结果调整

    # 清理测试文件
    import os
    try:
        os.remove(test_file_path)
        logger.info(f"已清理测试文件 {test_file_path}。")
    except OSError as e:
        logger.error(f"清理测试文件 {test_file_path} 失败: {e}")

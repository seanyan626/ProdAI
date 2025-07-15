import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, APP_NAME

def setup_logging():
    """
    配置应用的日志记录。
    """
    numeric_level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(numeric_level, int):
        logging.warning(f"无效的日志级别: {LOG_LEVEL}。将默认为 INFO。")
        numeric_level = logging.INFO

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # 移除任何现有的处理器以避免重复日志
    # 如果此函数可能被多次调用，或者其他库（如 FastAPI 中的 uvicorn）配置了日志记录，则此操作很重要。
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器 (可选)
    if LOG_FILE:
        try:
            file_handler = logging.FileHandler(LOG_FILE, mode='a') # 'a' 表示追加
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logging.info(f"日志将记录到文件: {LOG_FILE}")
        except Exception as e:
            logging.error(f"设置日志文件 {LOG_FILE} 失败: {e}", exc_info=True)

    # 控制特定库的日志记录 (例如，日志输出较多的库)
    # logging.getLogger("httpx").setLevel(logging.WARNING) # openai 使用的 httpx 示例
    # logging.getLogger("openai").setLevel(logging.WARNING) # openai 库示例

    logging.info(f"应用 {APP_NAME} 的日志记录已初始化，级别为 {LOG_LEVEL}。")

if __name__ == "__main__":
    # 这部分用于测试日志设置
    # 它需要 config.py 能够加载 LOG_LEVEL 等。
    # 要直接运行此文件，你可能需要调整 Python 的路径或作为模块运行。
    # 为简单起见，假设 config.py 可访问。

    # 首先，确保加载配置 (因为 setup_logging 依赖它)
    from .config import load_config
    load_config()

    # 现在设置日志记录
    setup_logging()

    # 测试日志记录
    root_logger = logging.getLogger()
    root_logger.debug("这是一条调试信息。")
    root_logger.info("这是一条普通信息。")
    root_logger.warning("这是一条警告信息。")
    root_logger.error("这是一条错误信息。")
    root_logger.critical("这是一条严重错误信息。")

    # 测试来自特定模块的日志记录
    module_logger = logging.getLogger("my_module") # 测试模块日志
    module_logger.info("来自 my_module 的普通信息。")

    if LOG_FILE:
        print(f"请检查日志文件: {LOG_FILE} (如果已配置且可写)")

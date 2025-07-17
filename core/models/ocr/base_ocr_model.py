# core/models/ocr/base_ocr_model.py
# OCR (光学字符识别) 模型的抽象基类
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """OCR基础异常"""
    pass


class OCRResult:
    """
    用于封装OCR识别结果的数据类。
    包含完整文本、文本段信息和置信度等详细信息。
    """

    def __init__(self, full_text: str, segments: Optional[List[Dict[str, Any]]] = None, confidence: float = 0.0):
        self.full_text: str = full_text  # 识别出的完整文本
        self.segments: List[Dict[str, Any]] = segments or []  # 文本段列表，每个段包含 'text', 'confidence', 'bbox' 等信息
        self.confidence: float = confidence  # 整体置信度

    def __repr__(self):
        return f"OCRResult(full_text='{self.full_text[:100]}...', segments_count={len(self.segments)}, confidence={self.confidence})"


class BaseOCRModel(ABC):
    """
    OCR (光学字符识别) 模型的抽象基类。
    子类应实现与特定 OCR 引擎或服务交互的具体逻辑。
    """

    def __init__(self, model_name: Optional[str] = None, max_retries: int = 3, timeout: int = 30, **kwargs):
        """
        初始化 OCR 模型。

        参数:
            model_name (Optional[str]): 要使用的 OCR 模型的名称或标识。
            max_retries (int): 最大重试次数。
            timeout (int): 请求超时时间（秒）。
            **kwargs: 其他特定于模型的配置参数。
        """
        self.model_name = model_name or "default_ocr"
        self.max_retries = max_retries
        self.timeout = timeout
        self.config = kwargs
        logger.info(f"OCR模型基类 '{self.__class__.__name__}' 使用模型 '{self.model_name}' 初始化。")

    @abstractmethod
    def recognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        从图像中识别文本。

        参数:
            image_source (Union[str, bytes]): 图像的来源。
                - str: 图像文件的路径。
                - bytes: 图像文件的字节数据。

        返回:
            OCRResult: 包含识别出的文本和可能的其他信息的 OCRResult 对象。
        """
        pass

    @abstractmethod
    async def arecognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        异步从图像中识别文本。

        参数:
            image_source (Union[str, bytes]): 图像的来源。

        返回:
            OCRResult: 包含识别出的文本和可能的其他信息的 OCRResult 对象。
        """
        pass

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        带有指数退避的重试机制。

        参数:
            func: 要重试的函数
            *args: 函数的位置参数
            **kwargs: 函数的关键字参数

        返回:
            函数的返回值

        抛出:
            OCRError: 当所有重试都失败时
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s, ...
                    logger.warning(f"OCR调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)}. {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OCR调用在 {self.max_retries + 1} 次尝试后仍然失败: {str(e)}")
        
        raise OCRError(f"OCR识别失败，已重试 {self.max_retries} 次: {str(last_exception)}")

    async def _async_retry_with_backoff(self, func, *args, **kwargs):
        """
        异步版本的带有指数退避的重试机制。
        
        使用场景:
            1. 异步API调用：当OCR服务使用异步HTTP客户端（如aiohttp）调用外部API
            2. 异步Web应用：在FastAPI或aiohttp等异步Web框架中使用OCR功能
            3. 并发处理：需要同时处理多个OCR请求而不阻塞主线程
            4. 异步工作流：作为异步数据处理管道的一部分

        参数:
            func: 要重试的异步函数
            *args: 函数的位置参数
            **kwargs: 函数的关键字参数

        返回:
            函数的返回值

        抛出:
            OCRError: 当所有重试都失败时
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # 异步等待函数执行完成并返回结果
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # 计算指数退避等待时间：1秒、2秒、4秒、8秒...
                    wait_time = 2 ** attempt  # 指数退避策略，避免对服务器造成压力
                    logger.warning(f"异步OCR调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {str(e)}. {wait_time}秒后重试...")
                    # 非阻塞等待，不会阻塞事件循环中的其他任务
                    await asyncio.sleep(wait_time)  # 使用异步sleep而不是time.sleep，避免阻塞事件循环
                else:
                    # 所有重试都失败，记录最终错误
                    logger.error(f"异步OCR调用在 {self.max_retries + 1} 次尝试后仍然失败: {str(e)}")
        
        # 抛出自定义异常，包含详细的错误信息
        raise OCRError(f"异步OCR识别失败，已重试 {self.max_retries} 次: {str(last_exception)}")

    def get_model_info(self) -> dict:
        """
        返回关于 OCR 模型的信息。
        """
        return {
            "model_name": self.model_name,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "config": self.config,
            "type": "OCRModel"
        }


if __name__ == '__main__':
    logger.info("BaseOCRModel 模块。这是一个抽象基类，不应直接实例化。OCRResult 类已定义。")
    # 示例：
    # class MyOCRModel(BaseOCRModel):
    #     def recognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
    #         if isinstance(image_source, str):
    #             logger.info(f"正在从路径 '{image_source}' 识别文本...")
    #         else:
    #             logger.info(f"正在从字节数据 (长度: {len(image_source)}) 识别文本...")
    #         # 模拟实现
    #         simulated_text = "这是从图像中识别出的模拟文本。"
    #         simulated_segments = [
    #             {"text": "这是", "confidence": 0.95, "bbox": [10, 10, 50, 30]},
    #             {"text": "模拟文本", "confidence": 0.90, "bbox": [10, 40, 100, 60]},
    #         ]
    #         return OCRResult(full_text=simulated_text, segments=simulated_segments)
    #
    #     async def arecognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
    #         # 简单地包装同步方法作为异步示例
    #         import asyncio
    #         return await asyncio.to_thread(self.recognize_text, image_source)
    #
    # # my_ocr_model = MyOCRModel(model_name="my-test-ocr-v1")
    # # print(my_ocr_model.get_model_info())
    # # dummy_image_path = "path/to/dummy_image.png" # 需要一个真实或模拟的路径/字节数据
    # # try:
    # #     with open(dummy_image_path, "rb") as f: dummy_bytes = f.read()
    # #     ocr_result_bytes = my_ocr_model.recognize_text(dummy_bytes)
    # #     print(f"OCR (bytes) 结果: {ocr_result_bytes}")
    # # except FileNotFoundError:
    # #     print(f"示例图片 {dummy_image_path} 未找到，跳过字节读取测试。")
    # # ocr_result_path = my_ocr_model.recognize_text(dummy_image_path) # 这会因文件不存在而失败
    # # print(f"OCR (path) 结果: {ocr_result_path}")
    pass

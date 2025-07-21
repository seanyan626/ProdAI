# core/models/ocr/cloud/dashscope_ocr_model.py
"""
DashScope OCR模型实现

使用阿里云百炼（DashScope）的OCR服务进行文字识别。
支持通用文字识别、表格识别等多种场景。
"""

import base64
import logging
from pathlib import Path
from typing import Union, Optional

from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse

from configs.config import config
from ..base_ocr_model import BaseOCRModel, OCRResult, OCRError

logger = logging.getLogger(__name__)


class DashScopeOCRModel(BaseOCRModel):
    """
    DashScope OCR模型实现
    
    使用阿里云百炼（DashScope）的多模态对话服务进行文字识别。
    支持通用文字识别功能。
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = "dashscope_ocr",
                 model: Optional[str] = None,
                 **kwargs):
        """
        初始化DashScope OCR模型
        
        参数:
            api_key: DashScope API密钥，如果不提供则从配置中获取
            model_name: 模型名称
            model: 使用的多模态模型，默认从配置中获取
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            max_retries=kwargs.get('max_retries', config.ocr.max_retries),
            timeout=kwargs.get('timeout', config.ocr.timeout),
            **kwargs
        )

        self.api_key = api_key or config.dashscope.api_key
        self.model = model or config.dashscope.ocr_model_name

        if not self.api_key:
            raise OCRError(
                "DashScope API密钥未配置。请在环境变量中设置 DASHSCOPE_API_KEY 或在初始化时提供 api_key 参数。")

        logger.info(f"DashScope OCR模型已初始化，使用模型: {self.model}")

    def _prepare_image_data(self, image_source: Union[str, bytes]) -> str:
        """
        准备图像数据，转换为DashScope多模态对话格式
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            str: 图像文件路径或base64编码的图像数据
        """
        try:
            if isinstance(image_source, str):
                # 如果是文件路径，检查文件是否存在
                if not Path(image_source).exists():
                    raise OCRError(f"图像文件未找到: {image_source}")
                return f"file://{Path(image_source).absolute()}"

            elif isinstance(image_source, bytes):
                # 如果是字节数据，转换为base64编码
                image_base64 = base64.b64encode(image_source).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"

            else:
                raise OCRError(f"不支持的图像数据类型: {type(image_source)}")

        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"准备图像数据失败: {str(e)}")

    def _parse_response(self, response: DashScopeAPIResponse) -> OCRResult:
        """
        解析DashScope多模态对话响应
        
        参数:
            response: DashScope API响应
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            # 检查响应状态
            if response.status_code != 200:
                error_msg = response.message or "未知错误"
                raise OCRError(f"DashScope API调用失败: {error_msg}")

            # 获取识别结果
            output = response.output
            if not output or 'choices' not in output:
                raise OCRError("响应中没有找到识别结果")

            # 提取文本内容
            choices = output['choices']
            if not choices:
                return OCRResult(full_text="", segments=[], confidence=0.0)

            # 获取第一个选择的消息内容
            message = choices[0].get('message', {})
            content = message.get('content', '')

            if not content:
                return OCRResult(full_text="", segments=[], confidence=0.0)

            # 简单解析：将整个内容作为一个段落
            segments = [{
                "text": content,
                "confidence": 0.9,  # DashScope不提供置信度，使用默认值
                "bbox": []
            }]

            return OCRResult(
                full_text=content,
                segments=segments,
                confidence=0.9
            )

        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"解析DashScope响应失败: {str(e)}")

    def recognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        同步识别图像中的文字
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            OCRResult: 识别结果
        """

        def _recognize():
            # 准备图像数据
            image_data = self._prepare_image_data(image_source)

            # 构建多模态对话消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},
                        {"text": "请识别图片中的所有文字内容，直接输出文字内容，不需要其他说明。"}
                    ]
                }
            ]

            # 发送API请求
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key
            )

            # 解析响应
            return self._parse_response(response)

        # 使用重试机制
        return self._retry_with_backoff(_recognize)

    async def arecognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        异步识别图像中的文字
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            OCRResult: 识别结果
        """

        async def _arecognize():
            # 准备图像数据
            image_data = self._prepare_image_data(image_source)

            # 构建多模态对话消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data},
                        {"text": "请识别图片中的所有文字内容，直接输出文字内容，不需要其他说明。"}
                    ]
                }
            ]

            # 发送异步API请求
            response = await MultiModalConversation.acall(
                model=self.model,
                messages=messages,
                api_key=self.api_key
            )

            # 解析响应
            return self._parse_response(response)

        # 使用异步重试机制
        return await self._async_retry_with_backoff(_arecognize)


if __name__ == "__main__":
    # 测试代码
    from configs.config import load_config
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()

    logger.info("DashScope OCR模型测试")

    try:
        # 创建DashScope OCR模型实例
        ocr = DashScopeOCRModel()
        logger.info(f"模型信息: {ocr.get_model_info()}")

        # 注意：实际测试需要有效的API密钥和图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")

    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")

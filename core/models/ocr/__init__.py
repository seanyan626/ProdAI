# core/models/ocr/__init__.py
"""
OCR (光学字符识别) 模型模块

该模块提供了统一的OCR接口和多种OCR服务的实现。
支持云服务OCR和本地OCR模型。
"""

from .base_ocr_model import BaseOCRModel, OCRResult, OCRError
from .openai_ocr_model import OpenAIOCRModel
from .ocr_factory import OCRFactory

# 导入子模块
from . import cloud
from . import local

# 导入具体的OCR模型
try:
    from .cloud.aliyun_ocr_model import AliyunOCRModel
except ImportError:
    AliyunOCRModel = None

__all__ = [
    "BaseOCRModel",
    "OCRResult", 
    "OCRError",
    "OpenAIOCRModel",
    "OCRFactory",
    "cloud",
    "local"
]

# 如果成功导入，添加到 __all__
if AliyunOCRModel:
    __all__.append("AliyunOCRModel")
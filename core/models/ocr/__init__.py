# core/models/ocr/__init__.py
"""
OCR (光学字符识别) 模型模块

该模块提供了统一的OCR接口和多种OCR服务的实现。
支持云服务OCR和本地OCR模型。
"""

from .base_ocr_model import BaseOCRModel, OCRResult, OCRError
from .openai_ocr_model import OpenAIOCRModel
# OCRFactory 按需导入，避免触发自动注册

# 导入子模块
from . import cloud
from . import local

# 导入具体的OCR模型
try:
    from .cloud.dashscope_ocr_model import DashScopeOCRModel
except ImportError:
    DashScopeOCRModel = None

__all__ = [
    "BaseOCRModel",
    "OCRResult", 
    "OCRError",
    "OpenAIOCRModel",
    "cloud",
    "local"
]

# 如果成功导入，添加到 __all__
if DashScopeOCRModel:
    __all__.append("DashScopeOCRModel")
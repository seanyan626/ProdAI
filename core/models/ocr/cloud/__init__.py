# core/models/ocr/cloud/__init__.py
"""
云服务OCR模型模块

该模块包含各种云服务OCR提供商的实现，如阿里云、百度等。
"""

from .aliyun_ocr_model import AliyunOCRModel
from .baidu_ocr_model import BaiduOCRModel

__all__ = [
    "AliyunOCRModel",
    "BaiduOCRModel"
]
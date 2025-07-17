# core/models/ocr/local/__init__.py
"""
本地OCR模型模块

该模块包含各种本地OCR引擎的实现，如PaddleOCR、EasyOCR等。
"""

# 导入具体的本地OCR模型
from .paddle_ocr_model import PaddleOCRModel
from .easy_ocr_model import EasyOCRModel

__all__ = [
    "PaddleOCRModel",
    "EasyOCRModel"
]
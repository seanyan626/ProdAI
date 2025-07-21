# core/models/ocr/ocr_factory.py
"""
OCR工厂类，用于创建不同类型的OCR模型实例。
"""

import logging
from typing import Dict, Any, Optional, Type

from .base_ocr_model import BaseOCRModel, OCRError
from .openai_ocr_model import OpenAIOCRModel

logger = logging.getLogger(__name__)


class OCRFactory:
    """
    OCR模型工厂类，用于创建不同类型的OCR模型实例。
    支持云服务OCR和本地OCR模型。
    """
    
    # 注册的OCR模型类
    _registered_models: Dict[str, Type[BaseOCRModel]] = {
        "openai": OpenAIOCRModel,
    }
    
    @classmethod
    def register_model(cls, provider_name: str, model_class: Type[BaseOCRModel]) -> None:
        """
        注册新的OCR模型类
        
        参数:
            provider_name: OCR提供商名称
            model_class: OCR模型类，必须是BaseOCRModel的子类
        """
        if not issubclass(model_class, BaseOCRModel):
            raise TypeError(f"模型类 {model_class.__name__} 必须是 BaseOCRModel 的子类")
        
        cls._registered_models[provider_name] = model_class
        logger.debug(f"已注册OCR模型: {provider_name} -> {model_class.__name__}")
    
    @classmethod
    def create_ocr(cls, provider: str, **kwargs) -> BaseOCRModel:
        """
        创建指定提供商的OCR模型实例
        
        参数:
            provider: OCR提供商名称，如 'dashscope', 'aliyun', 'baidu', 'paddle', 'easyocr', 'openai'
            **kwargs: 传递给OCR模型构造函数的参数
            
        返回:
            BaseOCRModel: OCR模型实例
            
        抛出:
            OCRError: 如果指定的提供商不支持
        """
        # 使用延迟加载获取模型类
        model_class = cls._lazy_load_model(provider.lower())
        
        # 创建并返回模型实例
        try:
            model_instance = model_class(**kwargs)
            logger.info(f"已创建OCR模型: {provider} ({model_class.__name__})")
            return model_instance
        except Exception as e:
            logger.error(f"创建OCR模型 {provider} 失败: {str(e)}")
            raise OCRError(f"创建OCR模型 {provider} 失败: {str(e)}") from e
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """
        获取所有支持的OCR提供商列表
        
        返回:
            list: 支持的OCR提供商名称列表
        """
        return list(cls._registered_models.keys())


    # 延迟加载的模型映射
    _model_mappings = {
        "dashscope": ("core.models.ocr.cloud.dashscope_ocr_model", "DashScopeOCRModel"),
        "aliyun": ("core.models.ocr.cloud.dashscope_ocr_model", "DashScopeOCRModel"),  # 向后兼容
        "baidu": ("core.models.ocr.cloud.baidu_ocr_model", "BaiduOCRModel"),
        "tencent": ("core.models.ocr.cloud.tencent_ocr_model", "TencentOCRModel"),
        "paddle": ("core.models.ocr.local.paddle_ocr_model", "PaddleOCRModel"),
        "easyocr": ("core.models.ocr.local.easy_ocr_model", "EasyOCRModel"),
    }
    
    @classmethod
    def _lazy_load_model(cls, provider: str) -> Type[BaseOCRModel]:
        """
        延迟加载OCR模型类
        
        参数:
            provider: OCR提供商名称
            
        返回:
            Type[BaseOCRModel]: OCR模型类
            
        抛出:
            OCRError: 如果模型无法加载
        """
        if provider in cls._registered_models:
            return cls._registered_models[provider]
        
        if provider not in cls._model_mappings:
            supported = ", ".join(list(cls._model_mappings.keys()) + list(cls._registered_models.keys()))
            raise OCRError(f"不支持的OCR提供商: {provider}。支持的提供商: {supported}")
        
        module_path, class_name = cls._model_mappings[provider]
        
        try:
            # 动态导入模块
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # 注册到缓存中，避免重复导入
            cls._registered_models[provider] = model_class
            logger.debug(f"延迟加载OCR模型: {provider} -> {class_name}")
            
            return model_class
            
        except ImportError as e:
            raise OCRError(f"无法导入OCR模型 {provider}: {str(e)}")
        except AttributeError as e:
            raise OCRError(f"OCR模型类 {class_name} 不存在: {str(e)}")


if __name__ == "__main__":
    # 示例用法
    try:
        # 获取支持的提供商列表
        providers = OCRFactory.get_supported_providers()
        print(f"支持的OCR提供商: {providers}")
        
        # 创建OpenAI OCR模型实例
        ocr = OCRFactory.create_ocr("openai", api_key="your_api_key_here")
        print(f"创建的OCR模型: {ocr.get_model_info()}")
        
        # 尝试创建不支持的提供商
        # ocr = OCRFactory.create_ocr("unsupported_provider")  # 这会抛出OCRError
    except OCRError as e:
        print(f"OCR错误: {str(e)}")
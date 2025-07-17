#!/usr/bin/env python3
"""
阿里云OCR模型使用示例

本示例展示如何使用阿里云OCR模型进行文字识别。
"""

import sys
import os
import base64

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import load_config
from configs.logging_config import setup_logging
from core.models.ocr.cloud.aliyun_ocr_model import AliyunOCRModel
from core.models.ocr.ocr_factory import OCRFactory
from core.models.ocr.base_ocr_model import OCRError
import logging

logger = logging.getLogger(__name__)


def example_direct_usage():
    """直接使用阿里云OCR模型的示例"""
    print("=== 直接使用阿里云OCR模型 ===")
    
    try:
        # 创建阿里云OCR模型实例
        # 注意：实际使用时需要提供真实的API密钥
        ocr = AliyunOCRModel(
            api_key="your_aliyun_ocr_api_key_here",  # 替换为真实的API密钥
            scene="general",  # 通用文字识别
            max_retries=3,
            timeout=30
        )
        
        print(f"模型信息: {ocr.get_model_info()}")
        
        # 模拟图像数据（实际使用时应该是真实的图像文件或字节数据）
        sample_image_data = b"fake_image_data_for_demo"
        
        print(f"准备识别图像数据（{len(sample_image_data)} 字节）...")
        
        # 注意：这里会因为使用假的API密钥和图像数据而失败
        # 在实际使用中，请提供真实的API密钥和图像数据
        # result = ocr.recognize_text(sample_image_data)
        # print(f"识别结果: {result.full_text}")
        
        print("✓ 模型创建成功（需要真实API密钥才能进行实际识别）")
        
    except OCRError as e:
        print(f"OCR错误: {str(e)}")
    except Exception as e:
        print(f"其他错误: {str(e)}")


def example_factory_usage():
    """通过工厂模式使用阿里云OCR模型的示例"""
    print("\n=== 通过工厂模式使用阿里云OCR模型 ===")
    
    try:
        # 获取支持的OCR提供商
        providers = OCRFactory.get_supported_providers()
        print(f"支持的OCR提供商: {providers}")
        
        # 通过工厂创建阿里云OCR模型
        ocr = OCRFactory.create_ocr(
            provider="aliyun",
            api_key="your_aliyun_ocr_api_key_here",  # 替换为真实的API密钥
            scene="table",  # 表格识别
            max_retries=2,
            timeout=20
        )
        
        print(f"通过工厂创建的模型信息: {ocr.get_model_info()}")
        print(f"识别场景: {ocr.scene}")
        
        print("✓ 工厂模式创建成功")
        
    except OCRError as e:
        print(f"OCR错误: {str(e)}")
    except Exception as e:
        print(f"其他错误: {str(e)}")


def example_different_scenes():
    """展示不同识别场景的使用"""
    print("\n=== 不同识别场景示例 ===")
    
    scenes = [
        ("general", "通用文字识别"),
        ("table", "表格识别"),
        ("handwriting", "手写文字识别")
    ]
    
    for scene, description in scenes:
        try:
            print(f"\n{description} ({scene}):")
            ocr = AliyunOCRModel(
                api_key="demo_api_key",
                scene=scene
            )
            print(f"  ✓ {description}模型创建成功")
            print(f"  API端点: {ocr.api_url}")
            
        except Exception as e:
            print(f"  ✗ {description}模型创建失败: {str(e)}")


def example_error_handling():
    """展示错误处理"""
    print("\n=== 错误处理示例 ===")
    
    # 1. 测试没有API密钥的情况
    print("1. 测试没有API密钥:")
    try:
        ocr = AliyunOCRModel()  # 没有提供API密钥
        print("  ✗ 应该抛出错误")
    except OCRError as e:
        print(f"  ✓ 正确捕获错误: {str(e)}")
    
    # 2. 测试不支持的提供商
    print("\n2. 测试不支持的OCR提供商:")
    try:
        ocr = OCRFactory.create_ocr("unsupported_provider")
        print("  ✗ 应该抛出错误")
    except OCRError as e:
        print(f"  ✓ 正确捕获错误: {str(e)}")


def example_async_usage():
    """展示异步使用方式"""
    print("\n=== 异步使用示例 ===")
    
    import asyncio
    
    async def async_ocr_demo():
        try:
            ocr = AliyunOCRModel(api_key="demo_api_key")
            print("异步OCR模型创建成功")
            
            # 注意：实际使用时需要真实的图像数据和API密钥
            # sample_data = b"fake_image_data"
            # result = await ocr.arecognize_text(sample_data)
            # print(f"异步识别结果: {result.full_text}")
            
            print("✓ 异步模型准备就绪（需要真实数据进行测试）")
            
        except Exception as e:
            print(f"异步示例错误: {str(e)}")
    
    # 运行异步示例
    asyncio.run(async_ocr_demo())


def main():
    """主函数"""
    print("阿里云OCR模型使用示例")
    print("=" * 50)
    
    # 加载配置
    load_config()
    setup_logging()
    
    # 运行各种示例
    example_direct_usage()
    example_factory_usage()
    example_different_scenes()
    example_error_handling()
    example_async_usage()
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("\n使用说明:")
    print("1. 将 'your_aliyun_ocr_api_key_here' 替换为真实的阿里云OCR API密钥")
    print("2. 提供真实的图像文件路径或图像字节数据")
    print("3. 根据需要选择合适的识别场景（general/table/handwriting）")
    print("4. 可以通过环境变量 ALIYUN_OCR_API_KEY 设置API密钥")


if __name__ == "__main__":
    main()
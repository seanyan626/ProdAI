#!/usr/bin/env python3
"""
DashScope OCR 使用示例

演示如何使用DashScope OCR模型进行文字识别。
包括基本使用、错误处理、异步调用等功能。
"""

import asyncio
import logging
from pathlib import Path

# 导入配置和日志
from configs.config import load_config
from configs.logging_config import setup_logging
from core.models.ocr.cloud.dashscope_ocr_model import DashScopeOCRModel
from core.models.ocr.ocr_factory import OCRFactory
from core.models.ocr.base_ocr_model import OCRError


def basic_ocr_example():
    """基本OCR使用示例"""
    print("=== 基本DashScope OCR使用示例 ===")
    
    try:
        # 创建DashScope OCR模型实例
        # 注意：实际使用时需要提供真实的API密钥
        ocr = DashScopeOCRModel(
            api_key="your_dashscope_api_key_here",  # 替换为真实的API密钥
            model="qwen-vl-max",  # 使用的多模态模型
            max_retries=3,
            timeout=30
        )
        
        print(f"模型信息: {ocr.get_model_info()}")
        
        # 测试图像路径（使用data/images目录下的图像文件）
        test_image_path = "data/images/1.png"
        
        if Path(test_image_path).exists():
            print(f"正在识别图像: {test_image_path}")
            result = ocr.recognize_text(test_image_path)
            
            print(f"识别结果:")
            print(f"  完整文本: {result.full_text}")
            print(f"  置信度: {result.confidence}")
            print(f"  文本段数量: {len(result.segments)}")
            
            for i, segment in enumerate(result.segments):
                print(f"  段落 {i+1}: {segment['text']} (置信度: {segment['confidence']})")
        else:
            print(f"测试图像文件 {test_image_path} 不存在")
            print("请提供有效的图像文件路径进行测试")
            
    except OCRError as e:
        print(f"OCR错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")


def factory_example():
    """使用工厂模式创建OCR实例"""
    print("\n=== 使用工厂模式创建DashScope OCR ===")
    
    try:
        # 使用工厂创建DashScope OCR实例
        ocr = OCRFactory.create_ocr(
            provider="dashscope",
            api_key="your_dashscope_api_key_here",  # 替换为真实的API密钥
            model="qwen-vl-max",
            max_retries=2,
            timeout=25
        )
        
        print(f"通过工厂创建的模型信息: {ocr.get_model_info()}")
        
    except OCRError as e:
        print(f"工厂创建OCR失败: {e}")
    except Exception as e:
        print(f"其他错误: {e}")


def error_handling_examples():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 测试各种错误情况
    test_cases = [
        ("没有API密钥", {}),
        ("无效的图像路径", {"api_key": "demo_key"}),
        ("无效的图像数据", {"api_key": "demo_key"})
    ]
    
    for description, kwargs in test_cases:
        try:
            print(f"\n{description}:")
            ocr = DashScopeOCRModel(**kwargs)
            
            if "图像路径" in description:
                result = ocr.recognize_text("nonexistent_image.jpg")
            elif "图像数据" in description:
                result = ocr.recognize_text(12345)  # 无效的数据类型
            else:
                print("  ✓ OCR实例创建成功")
                
        except OCRError as e:
            print(f"  ✓ 预期的OCR错误: {e}")
        except Exception as e:
            print(f"  ✗ 未预期的错误: {e}")


def test_without_api_key():
    """测试没有API密钥的情况"""
    print("\n=== 测试API密钥验证 ===")
    
    print("1. 测试没有API密钥:")
    try:
        ocr = DashScopeOCRModel()  # 没有提供API密钥
        print("  ✗ 应该抛出错误")
    except OCRError as e:
        print(f"  ✓ 正确抛出错误: {e}")
    
    print("\n2. 测试提供API密钥:")
    try:
        ocr = DashScopeOCRModel(api_key="test_key")
        print("  ✓ API密钥验证通过")
    except Exception as e:
        print(f"  ✗ 意外错误: {e}")


def bytes_data_example():
    """字节数据处理示例"""
    print("\n=== 字节数据处理示例 ===")
    
    try:
        ocr = DashScopeOCRModel(api_key="demo_key")
        
        # 模拟图像字节数据
        fake_image_bytes = b"fake image data for testing"
        
        print("准备处理字节数据...")
        image_data = ocr._prepare_image_data(fake_image_bytes)
        print(f"处理结果: {image_data[:50]}...")  # 只显示前50个字符
        
    except Exception as e:
        print(f"字节数据处理错误: {e}")


async def async_ocr_example():
    """异步OCR示例"""
    print("\n=== 异步OCR示例 ===")
    
    try:
        ocr = DashScopeOCRModel(api_key="demo_api_key")
        print("异步OCR模型创建成功")
        
        # 注意：这里只是演示异步调用的语法
        # 使用data/images目录下的图像文件
        test_image = "data/images/1.png"
        
        if Path(test_image).exists():
            print(f"开始异步识别: {test_image}")
            result = await ocr.arecognize_text(test_image)
            print(f"异步识别结果: {result.full_text}")
        else:
            print("异步OCR准备就绪，等待真实图像文件")
            
    except Exception as e:
        print(f"异步OCR错误: {e}")


def main():
    """主函数"""
    # 加载配置和设置日志
    load_config()
    setup_logging()
    
    print("DashScope OCR 使用示例")
    print("=" * 50)
    
    # 运行各种示例
    basic_ocr_example()
    factory_example()
    error_handling_examples()
    test_without_api_key()
    bytes_data_example()
    
    # 运行异步示例
    print("\n运行异步示例...")
    asyncio.run(async_ocr_example())
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("\n使用说明:")
    print("1. 将 'your_dashscope_api_key_here' 替换为真实的DashScope API密钥")
    print("2. 提供真实的图像文件路径或图像字节数据")
    print("3. 根据需要选择合适的多模态模型（如 qwen-vl-max）")
    print("4. 可以通过环境变量 DASHSCOPE_API_KEY 设置API密钥")
    print("5. 支持同步和异步两种调用方式")


if __name__ == "__main__":
    main()
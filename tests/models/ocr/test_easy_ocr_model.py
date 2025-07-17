import os
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import tempfile

from core.models.ocr.local.easy_ocr_model import EasyOCRModel, OCRError, EASYOCR_AVAILABLE

# 跳过测试如果EasyOCR不可用
pytestmark = pytest.mark.skipif(not EASYOCR_AVAILABLE, reason="EasyOCR not installed")


def create_test_image(text="Hello World", size=(200, 100), color=(0, 0, 0), bg_color=(255, 255, 255)):
    """创建一个包含文本的测试图像"""
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        # 尝试使用系统字体
        font_size = 20
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()
    
    # 计算文本位置使其居中
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # 绘制文本
    draw.text(position, text, fill=color, font=font)
    
    return image


class TestEasyOCRModel:
    """EasyOCR模型测试类"""
    
    @classmethod
    def setup_class(cls):
        """设置测试环境"""
        # 创建一个临时目录用于存储测试图像
        cls.temp_dir = tempfile.mkdtemp()
        
        # 创建测试图像
        cls.test_image = create_test_image(text="Hello World")
        cls.test_image_path = os.path.join(cls.temp_dir, "test_image.png")
        cls.test_image.save(cls.test_image_path)
        
        # 创建中文测试图像
        cls.test_image_zh = create_test_image(text="你好世界")
        cls.test_image_zh_path = os.path.join(cls.temp_dir, "test_image_zh.png")
        cls.test_image_zh.save(cls.test_image_zh_path)
        
        # 将图像转换为字节
        img_byte_arr = io.BytesIO()
        cls.test_image.save(img_byte_arr, format='PNG')
        cls.test_image_bytes = img_byte_arr.getvalue()
    
    @classmethod
    def teardown_class(cls):
        """清理测试环境"""
        # 删除临时文件
        if os.path.exists(cls.test_image_path):
            os.remove(cls.test_image_path)
        if os.path.exists(cls.test_image_zh_path):
            os.remove(cls.test_image_zh_path)
        
        # 删除临时目录
        os.rmdir(cls.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        # 基本初始化
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        assert ocr.languages == ['en']
        assert ocr.gpu is False
        
        # 测试模型信息
        model_info = ocr.get_model_info()
        assert model_info['model_name'] == 'easyocr'
        assert model_info['languages'] == ['en']
        assert model_info['gpu'] is False
        assert 'supported_languages' in model_info
        assert 'en' in model_info['supported_languages']
    
    def test_get_supported_languages(self):
        """测试获取支持的语言"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        languages = ocr.get_supported_languages()
        
        # 检查常见语言是否存在
        assert 'en' in languages
        assert 'ch_sim' in languages
        assert 'ja' in languages
        assert languages['en'] == 'English'
        assert languages['ch_sim'] == 'Chinese (Simplified)'
    
    @pytest.mark.slow
    def test_recognize_text_from_path(self):
        """测试从文件路径识别文本"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        result = ocr.recognize_text(self.test_image_path)
        
        # 检查结果
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0
        assert "Hello" in result.full_text or "HELLO" in result.full_text.upper()
        assert len(result.segments) > 0
        assert result.confidence > 0
    
    @pytest.mark.slow
    def test_recognize_text_from_bytes(self):
        """测试从字节数据识别文本"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        result = ocr.recognize_text(self.test_image_bytes)
        
        # 检查结果
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0
        assert "Hello" in result.full_text or "HELLO" in result.full_text.upper()
        assert len(result.segments) > 0
        assert result.confidence > 0
    
    @pytest.mark.slow
    def test_recognize_chinese_text(self):
        """测试识别中文文本"""
        ocr = EasyOCRModel(languages=['ch_sim', 'en'], gpu=False)
        result = ocr.recognize_text(self.test_image_zh_path)
        
        # 检查结果
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0
        # 由于OCR识别可能不完全准确，我们只检查结果是否包含部分中文字符
        assert any(c in result.full_text for c in "你好世界")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_arecognize_text(self):
        """测试异步识别文本"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        result = await ocr.arecognize_text(self.test_image_path)
        
        # 检查结果
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0
        assert "Hello" in result.full_text or "HELLO" in result.full_text.upper()
        assert len(result.segments) > 0
        assert result.confidence > 0
    
    def test_invalid_image_path(self):
        """测试无效的图像路径"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        
        with pytest.raises(OCRError):
            ocr.recognize_text("non_existent_image.jpg")
    
    def test_invalid_image_data(self):
        """测试无效的图像数据"""
        ocr = EasyOCRModel(languages=['en'], gpu=False)
        
        with pytest.raises(OCRError):
            ocr.recognize_text(b"invalid image data")
    
    @pytest.mark.slow
    def test_auto_language_detection(self):
        """测试自动语言检测"""
        # 创建启用自动语言检测的OCR模型
        ocr = EasyOCRModel(languages=['en'], gpu=False, auto_language_detection=True)
        
        # 识别中文图像
        result = ocr.recognize_text(self.test_image_zh_path)
        
        # 检查结果
        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0
        # 由于OCR识别可能不完全准确，我们只检查结果是否包含部分中文字符
        assert any(c in result.full_text for c in "你好世界")


if __name__ == "__main__":
    # 手动运行测试
    test = TestEasyOCRModel()
    test.setup_class()
    try:
        test.test_init()
        test.test_get_supported_languages()
        print("基本测试通过")
        
        # 注意：以下测试需要安装EasyOCR并下载模型，可能需要一些时间
        # test.test_recognize_text_from_path()
        # test.test_recognize_text_from_bytes()
        # test.test_recognize_chinese_text()
        # print("识别测试通过")
    finally:
        test.teardown_class()
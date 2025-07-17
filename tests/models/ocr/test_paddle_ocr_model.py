# tests/models/ocr/test_paddle_ocr_model.py
"""
PaddleOCR模型单元测试
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import io

from core.models.ocr.local.paddle_ocr_model import PaddleOCRModel, OCRResult, OCRError, PADDLEOCR_AVAILABLE


# 跳过所有测试，如果PaddleOCR不可用
pytestmark = pytest.mark.skipif(not PADDLEOCR_AVAILABLE, reason="PaddleOCR库未安装")


class TestPaddleOCRModel:
    """PaddleOCR模型测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 创建一个模拟的PaddleOCR实例
        self.mock_paddle_ocr = MagicMock()
        
        # 模拟识别结果
        self.mock_result = [
            [
                [[10, 10], [100, 10], [100, 30], [10, 30]],
                ["测试文本1", 0.95]
            ],
            [
                [[10, 40], [100, 40], [100, 60], [10, 60]],
                ["测试文本2", 0.90]
            ]
        ]
        
        # 设置模拟的OCR方法返回值
        self.mock_paddle_ocr.ocr.return_value = self.mock_result
        
        # 创建一个测试用的图像
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image_bytes = self._image_to_bytes(self.test_image)
        
        # 临时文件路径
        self.temp_image_path = "temp_test_image.png"
        
        # 保存测试图像
        Image.fromarray(self.test_image).save(self.temp_image_path)

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 删除临时文件
        if os.path.exists(self.temp_image_path):
            os.remove(self.temp_image_path)

    def _image_to_bytes(self, image):
        """将numpy图像转换为字节数据"""
        pil_image = Image.fromarray(image)
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='PNG')
        return byte_io.getvalue()

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_init(self, mock_paddle_ocr_class):
        """测试初始化"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 验证初始化参数
        assert ocr.lang == 'ch'
        assert ocr.use_gpu is False
        assert ocr.model_name == 'paddleocr'
        assert ocr._ocr is None  # 延迟初始化，应该为None
        
        # 获取OCR实例，触发初始化
        ocr._get_ocr()
        
        # 验证PaddleOCR类被正确调用
        mock_paddle_ocr_class.assert_called_once()
        assert ocr._ocr is not None

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_recognize_text_with_file_path(self, mock_paddle_ocr_class):
        """测试使用文件路径识别文本"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 识别文本
        result = ocr.recognize_text(self.temp_image_path)
        
        # 验证结果
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "测试文本1"
        assert result.segments[0]["confidence"] == 0.95
        assert result.segments[1]["text"] == "测试文本2"
        assert result.segments[1]["confidence"] == 0.90
        assert result.confidence == pytest.approx(0.925)  # (0.95 + 0.90) / 2
        
        # 验证PaddleOCR.ocr方法被正确调用
        self.mock_paddle_ocr.ocr.assert_called_once_with(
            img=self.temp_image_path,
            cls=True,
            det=True,
            rec=True
        )

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_recognize_text_with_bytes(self, mock_paddle_ocr_class):
        """测试使用字节数据识别文本"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 识别文本
        result = ocr.recognize_text(self.test_image_bytes)
        
        # 验证结果
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        
        # 验证PaddleOCR.ocr方法被调用
        self.mock_paddle_ocr.ocr.assert_called_once()

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_recognize_text_with_empty_result(self, mock_paddle_ocr_class):
        """测试空结果"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 设置空结果
        self.mock_paddle_ocr.ocr.return_value = []
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 识别文本
        result = ocr.recognize_text(self.temp_image_path)
        
        # 验证结果
        assert isinstance(result, OCRResult)
        assert result.full_text == ""
        assert len(result.segments) == 0
        assert result.confidence == 0.0

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_recognize_text_with_invalid_path(self, mock_paddle_ocr_class):
        """测试无效的文件路径"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 使用不存在的文件路径
        with pytest.raises(OCRError) as excinfo:
            ocr.recognize_text("non_existent_file.jpg")
        
        # 验证错误信息
        assert "图像文件不存在" in str(excinfo.value)

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_recognize_text_with_invalid_data_type(self, mock_paddle_ocr_class):
        """测试无效的数据类型"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 使用无效的数据类型
        with pytest.raises(OCRError) as excinfo:
            ocr.recognize_text(123)  # 整数不是有效的图像源
        
        # 验证错误信息
        assert "不支持的图像数据类型" in str(excinfo.value)

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    @pytest.mark.asyncio
    async def test_arecognize_text(self, mock_paddle_ocr_class):
        """测试异步识别文本"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 异步识别文本
        result = await ocr.arecognize_text(self.temp_image_path)
        
        # 验证结果
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        
        # 验证PaddleOCR.ocr方法被调用
        self.mock_paddle_ocr.ocr.assert_called_once()

    def test_get_supported_languages(self):
        """测试获取支持的语言列表"""
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 获取支持的语言
        languages = ocr.get_supported_languages()
        
        # 验证结果
        assert isinstance(languages, dict)
        assert 'ch' in languages
        assert 'en' in languages
        assert languages['ch'] == '中文'
        assert languages['en'] == '英文'

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    def test_get_model_info(self, mock_paddle_ocr_class):
        """测试获取模型信息"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        
        # 获取模型信息
        info = ocr.get_model_info()
        
        # 验证结果
        assert isinstance(info, dict)
        assert info['model_name'] == 'paddleocr'
        assert info['lang'] == 'ch'
        assert info['use_gpu'] is False
        assert info['type'] == 'LocalOCRModel'
        assert 'supported_languages' in info
        assert 'ch' in info['supported_languages']

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    @patch('core.models.ocr.local.paddle_ocr_model.os.path.exists')
    def test_check_model_files_with_missing_files(self, mock_exists, mock_paddle_ocr_class):
        """测试检查模型文件，文件不存在的情况"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 设置文件不存在
        mock_exists.return_value = False
        
        # 创建PaddleOCR模型实例，禁用下载
        ocr = PaddleOCRModel(lang='ch', use_gpu=False, download_enabled=False)
        
        # 检查模型文件，应该抛出异常
        with pytest.raises(OCRError) as excinfo:
            ocr._check_model_files()
        
        # 验证错误信息
        assert "不存在" in str(excinfo.value)
        assert "且不允许自动下载" in str(excinfo.value)

    @patch('core.models.ocr.local.paddle_ocr_model.PaddleOCR')
    @patch('core.models.ocr.local.paddle_ocr_model.os.path.exists')
    def test_check_model_files_with_existing_files(self, mock_exists, mock_paddle_ocr_class):
        """测试检查模型文件，文件存在的情况"""
        # 设置模拟的PaddleOCR类
        mock_paddle_ocr_class.return_value = self.mock_paddle_ocr
        
        # 设置文件存在
        mock_exists.return_value = True
        
        # 创建PaddleOCR模型实例，禁用下载
        ocr = PaddleOCRModel(lang='ch', use_gpu=False, download_enabled=False)
        
        # 检查模型文件，不应该抛出异常
        ocr._check_model_files()
        
        # 验证os.path.exists被调用
        assert mock_exists.called
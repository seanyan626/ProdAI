# tests/models/ocr/test_dashscope_ocr_model.py
"""
DashScope OCR模型的单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import base64
import asyncio

from core.models.ocr.cloud.dashscope_ocr_model import DashScopeOCRModel
from core.models.ocr.base_ocr_model import OCRResult, OCRError


class TestDashScopeOCRModel:
    """DashScope OCR模型测试类"""
    
    @pytest.fixture
    def mock_api_key(self):
        """模拟API密钥"""
        return "test_dashscope_api_key"
    
    @pytest.fixture
    def mock_response(self):
        """模拟DashScope API响应"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.output = {
            'choices': [
                {
                    'message': {
                        'content': '这是识别出的文字内容'
                    }
                }
            ]
        }
        return mock_resp
    
    @pytest.fixture
    def mock_error_response(self):
        """模拟错误响应"""
        mock_resp = Mock()
        mock_resp.status_code = 400
        mock_resp.message = "API调用失败"
        return mock_resp
    
    @pytest.fixture
    def mock_empty_response(self):
        """模拟空响应"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.output = {
            'choices': []
        }
        return mock_resp
    
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        """创建临时图像文件"""
        image_file = tmp_path / "test_image.jpg"
        image_file.write_bytes(b"fake image data")
        return str(image_file)
    
    @pytest.fixture
    def sample_image_bytes(self):
        """样本图像字节数据"""
        return b"fake image data"
    
    @pytest.fixture
    def ocr_model(self, mock_api_key):
        """创建DashScope OCR模型实例"""
        return DashScopeOCRModel(api_key=mock_api_key)
    
    def test_init_with_api_key(self, mock_api_key):
        """测试使用API密钥初始化"""
        ocr = DashScopeOCRModel(api_key=mock_api_key)
        assert ocr.api_key == mock_api_key
        assert ocr.model_name == "dashscope_ocr"
    
    def test_init_without_api_key(self):
        """测试没有API密钥时的初始化"""
        with patch('core.models.ocr.cloud.dashscope_ocr_model.config') as mock_config:
            mock_config.dashscope.api_key = None
            with pytest.raises(OCRError, match="DashScope API密钥未配置"):
                DashScopeOCRModel()
    
    def test_init_with_config_api_key(self):
        """测试从配置中获取API密钥"""
        with patch('core.models.ocr.cloud.dashscope_ocr_model.config') as mock_config:
            mock_config.dashscope.api_key = "config_api_key"
            mock_config.dashscope.ocr_model_name = "qwen-vl-max"
            mock_config.ocr.max_retries = 3
            mock_config.ocr.timeout = 30
            
            ocr = DashScopeOCRModel()
            assert ocr.api_key == "config_api_key"
            assert ocr.model == "qwen-vl-max"
    
    def test_prepare_image_data_with_path(self, ocr_model, sample_image_path):
        """测试准备图像数据 - 文件路径"""
        result = ocr_model._prepare_image_data(sample_image_path)
        assert result.startswith("file://")
        assert sample_image_path in result
    
    def test_prepare_image_data_with_bytes(self, ocr_model, sample_image_bytes):
        """测试准备图像数据 - 字节数据"""
        result = ocr_model._prepare_image_data(sample_image_bytes)
        assert result.startswith("data:image/jpeg;base64,")
        
        # 验证base64编码
        expected_base64 = base64.b64encode(sample_image_bytes).decode('utf-8')
        assert expected_base64 in result
    
    def test_prepare_image_data_with_nonexistent_file(self, ocr_model):
        """测试准备图像数据 - 不存在的文件"""
        with pytest.raises(OCRError, match="图像文件未找到"):
            ocr_model._prepare_image_data("nonexistent_file.jpg")
    
    def test_prepare_image_data_with_invalid_type(self, ocr_model):
        """测试准备图像数据 - 无效类型"""
        with pytest.raises(OCRError, match="不支持的图像数据类型"):
            ocr_model._prepare_image_data(123)
    
    def test_parse_response_success(self, ocr_model, mock_response):
        """测试解析成功响应"""
        result = ocr_model._parse_response(mock_response)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "这是识别出的文字内容"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "这是识别出的文字内容"
        assert result.segments[0]["confidence"] == 0.9
        assert result.confidence == 0.9
    
    def test_parse_response_error(self, ocr_model, mock_error_response):
        """测试解析错误响应"""
        with pytest.raises(OCRError, match="DashScope API调用失败"):
            ocr_model._parse_response(mock_error_response)
    
    def test_parse_response_empty(self, ocr_model, mock_empty_response):
        """测试解析空响应"""
        result = ocr_model._parse_response(mock_empty_response)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == ""
        assert len(result.segments) == 0
        assert result.confidence == 0.0
    
    def test_parse_response_no_output(self, ocr_model):
        """测试解析没有输出的响应"""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.output = None
        
        with pytest.raises(OCRError, match="响应中没有找到识别结果"):
            ocr_model._parse_response(mock_resp)
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    def test_recognize_text_success(self, mock_conversation, ocr_model, sample_image_path, mock_response):
        """测试同步文字识别成功"""
        mock_conversation.call.return_value = mock_response
        
        result = ocr_model.recognize_text(sample_image_path)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "这是识别出的文字内容"
        
        # 验证API调用参数
        mock_conversation.call.assert_called_once()
        call_args = mock_conversation.call.call_args
        assert call_args[1]['model'] == ocr_model.model
        assert call_args[1]['api_key'] == ocr_model.api_key
        assert 'messages' in call_args[1]
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    def test_recognize_text_with_bytes(self, mock_conversation, ocr_model, sample_image_bytes, mock_response):
        """测试使用字节数据进行文字识别"""
        mock_conversation.call.return_value = mock_response
        
        result = ocr_model.recognize_text(sample_image_bytes)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "这是识别出的文字内容"
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    def test_recognize_text_api_error(self, mock_conversation, ocr_model, sample_image_path, mock_error_response):
        """测试API调用错误"""
        mock_conversation.call.return_value = mock_error_response
        
        with pytest.raises(OCRError, match="DashScope API调用失败"):
            ocr_model.recognize_text(sample_image_path)
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    def test_recognize_text_with_retry(self, mock_conversation, ocr_model, sample_image_path, mock_response):
        """测试重试机制"""
        # 第一次调用失败，第二次成功
        mock_conversation.call.side_effect = [Exception("网络错误"), mock_response]
        
        result = ocr_model.recognize_text(sample_image_path)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "这是识别出的文字内容"
        assert mock_conversation.call.call_count == 2
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    @pytest.mark.asyncio
    async def test_arecognize_text_success(self, mock_conversation, ocr_model, sample_image_path, mock_response):
        """测试异步文字识别成功"""
        mock_conversation.acall.return_value = mock_response
        
        result = await ocr_model.arecognize_text(sample_image_path)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "这是识别出的文字内容"
        
        # 验证异步API调用
        mock_conversation.acall.assert_called_once()
    
    @patch('core.models.ocr.cloud.dashscope_ocr_model.MultiModalConversation')
    @pytest.mark.asyncio
    async def test_arecognize_text_error(self, mock_conversation, ocr_model, sample_image_path, mock_error_response):
        """测试异步API调用错误"""
        mock_conversation.acall.return_value = mock_error_response
        
        with pytest.raises(OCRError, match="DashScope API调用失败"):
            await ocr_model.arecognize_text(sample_image_path)
    
    def test_get_model_info(self, ocr_model):
        """测试获取模型信息"""
        info = ocr_model.get_model_info()
        
        assert "model_name" in info
        assert "provider" in info
        assert info["model_name"] == "dashscope_ocr"
        assert info["provider"] == "DashScope"


if __name__ == "__main__":
    pytest.main([__file__])
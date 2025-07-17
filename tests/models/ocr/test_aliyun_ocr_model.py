# tests/models/ocr/test_aliyun_ocr_model.py
"""
阿里云OCR模型的单元测试
"""

import pytest
import json
import base64
from unittest.mock import Mock, patch, mock_open
import asyncio

from core.models.ocr.cloud.aliyun_ocr_model import AliyunOCRModel
from core.models.ocr.base_ocr_model import OCRResult, OCRError


class TestAliyunOCRModel:
    """阿里云OCR模型测试类"""
    
    @pytest.fixture
    def mock_api_key(self):
        """模拟API密钥"""
        return "test_aliyun_api_key"
    
    @pytest.fixture
    def sample_image_bytes(self):
        """模拟图像字节数据"""
        return b"fake_image_data"
    
    @pytest.fixture
    def sample_image_base64(self, sample_image_bytes):
        """模拟base64编码的图像数据"""
        return base64.b64encode(sample_image_bytes).decode('utf-8')
    
    @pytest.fixture
    def mock_aliyun_response(self):
        """模拟阿里云OCR API响应"""
        return {
            "success": True,
            "data": {
                "content": [
                    {
                        "text": "测试文本1",
                        "prob": 0.95,
                        "text_rect": [[10, 10], [100, 10], [100, 30], [10, 30]]
                    },
                    {
                        "text": "测试文本2", 
                        "prob": 0.90,
                        "text_rect": [[10, 40], [120, 40], [120, 60], [10, 60]]
                    }
                ]
            }
        }
    
    @pytest.fixture
    def ocr_model(self, mock_api_key):
        """创建阿里云OCR模型实例"""
        return AliyunOCRModel(api_key=mock_api_key)
    
    def test_init_with_api_key(self, mock_api_key):
        """测试使用API密钥初始化"""
        ocr = AliyunOCRModel(api_key=mock_api_key)
        assert ocr.api_key == mock_api_key
        assert ocr.scene == "general"
        assert ocr.model_name == "aliyun_ocr"
    
    def test_init_without_api_key(self):
        """测试没有API密钥时的初始化"""
        with patch('core.models.ocr.cloud.aliyun_ocr_model.ALIYUN_OCR_API_KEY', None):
            with pytest.raises(OCRError, match="阿里云OCR API密钥未配置"):
                AliyunOCRModel()
    
    def test_init_with_config_api_key(self):
        """测试从配置获取API密钥"""
        with patch('core.models.ocr.cloud.aliyun_ocr_model.ALIYUN_OCR_API_KEY', 'config_api_key'):
            ocr = AliyunOCRModel()
            assert ocr.api_key == 'config_api_key'
    
    def test_init_with_different_scenes(self, mock_api_key):
        """测试不同识别场景的初始化"""
        # 测试表格识别
        ocr_table = AliyunOCRModel(api_key=mock_api_key, scene="table")
        assert ocr_table.scene == "table"
        
        # 测试手写识别
        ocr_handwriting = AliyunOCRModel(api_key=mock_api_key, scene="handwriting")
        assert ocr_handwriting.scene == "handwriting"
        
        # 测试不支持的场景，应该回退到通用识别
        ocr_unsupported = AliyunOCRModel(api_key=mock_api_key, scene="unsupported")
        assert ocr_unsupported.scene == "general"
    
    def test_prepare_image_data_from_bytes(self, ocr_model, sample_image_bytes, sample_image_base64):
        """测试从字节数据准备图像"""
        result = ocr_model._prepare_image_data(sample_image_bytes)
        assert result == sample_image_base64
    
    def test_prepare_image_data_from_file(self, ocr_model, sample_image_bytes, sample_image_base64):
        """测试从文件路径准备图像"""
        with patch("builtins.open", mock_open(read_data=sample_image_bytes)):
            result = ocr_model._prepare_image_data("test_image.jpg")
            assert result == sample_image_base64
    
    def test_prepare_image_data_file_not_found(self, ocr_model):
        """测试文件不存在的情况"""
        with pytest.raises(OCRError, match="图像文件未找到"):
            ocr_model._prepare_image_data("nonexistent_file.jpg")
    
    def test_prepare_image_data_invalid_type(self, ocr_model):
        """测试无效的图像数据类型"""
        with pytest.raises(OCRError, match="不支持的图像数据类型"):
            ocr_model._prepare_image_data(123)
    
    def test_parse_general_response(self, ocr_model, mock_aliyun_response):
        """测试解析通用识别响应"""
        result = ocr_model._parse_response(mock_aliyun_response)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "测试文本1"
        assert result.segments[0]["confidence"] == 0.95
        assert result.segments[0]["bbox"] == [10, 10, 100, 30]
        assert result.confidence == 0.925  # (0.95 + 0.90) / 2
    
    def test_parse_response_api_error(self, ocr_model):
        """测试API错误响应的解析"""
        error_response = {
            "success": False,
            "message": "API调用失败"
        }
        
        with pytest.raises(OCRError, match="阿里云OCR API调用失败: API调用失败"):
            ocr_model._parse_response(error_response)
    
    def test_parse_table_response(self, ocr_model):
        """测试解析表格识别响应"""
        table_response = {
            "success": True,
            "data": {
                "tables": [
                    {
                        "body": [
                            [
                                {"text": "单元格1", "confidence": 0.9, "bbox": [0, 0, 50, 20]},
                                {"text": "单元格2", "confidence": 0.8, "bbox": [50, 0, 100, 20]}
                            ]
                        ]
                    }
                ]
            }
        }
        
        ocr_model.scene = "table"
        result = ocr_model._parse_response(table_response)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "单元格1\n单元格2"
        assert len(result.segments) == 2
    
    @patch('requests.post')
    def test_make_request_success(self, mock_post, ocr_model, mock_aliyun_response, sample_image_base64):
        """测试成功的HTTP请求"""
        mock_response = Mock()
        mock_response.json.return_value = mock_aliyun_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = ocr_model._make_request(sample_image_base64)
        
        assert result == mock_aliyun_response
        mock_post.assert_called_once()
        
        # 验证请求参数
        call_args = mock_post.call_args
        assert call_args[1]['timeout'] == ocr_model.timeout
        assert 'Authorization' in call_args[1]['headers']
        assert 'image' in call_args[1]['json']
    
    @patch('requests.post')
    def test_make_request_timeout(self, mock_post, ocr_model, sample_image_base64):
        """测试请求超时"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(OCRError, match="阿里云OCR API请求超时"):
            ocr_model._make_request(sample_image_base64)
    
    @patch('requests.post')
    def test_make_request_http_error(self, mock_post, ocr_model, sample_image_base64):
        """测试HTTP错误"""
        import requests
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        with pytest.raises(OCRError, match="阿里云OCR API HTTP错误"):
            ocr_model._make_request(sample_image_base64)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_make_async_request_success(self, mock_post, ocr_model, mock_aliyun_response, sample_image_base64):
        """测试成功的异步HTTP请求"""
        mock_response = Mock()
        mock_response.json = Mock(return_value=asyncio.coroutine(lambda: mock_aliyun_response)())
        mock_response.raise_for_status = Mock()
        mock_post.return_value.__aenter__ = Mock(return_value=mock_response)
        mock_post.return_value.__aexit__ = Mock(return_value=None)
        
        result = await ocr_model._make_async_request(sample_image_base64)
        
        assert result == mock_aliyun_response
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_make_async_request_timeout(self, mock_post, ocr_model, sample_image_base64):
        """测试异步请求超时"""
        mock_post.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(OCRError, match="阿里云OCR API异步请求超时"):
            await ocr_model._make_async_request(sample_image_base64)
    
    @patch.object(AliyunOCRModel, '_make_request')
    @patch.object(AliyunOCRModel, '_prepare_image_data')
    def test_recognize_text_success(self, mock_prepare, mock_request, ocr_model, 
                                   sample_image_bytes, sample_image_base64, mock_aliyun_response):
        """测试成功的文字识别"""
        mock_prepare.return_value = sample_image_base64
        mock_request.return_value = mock_aliyun_response
        
        result = ocr_model.recognize_text(sample_image_bytes)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        mock_prepare.assert_called_once_with(sample_image_bytes)
        mock_request.assert_called_once_with(sample_image_base64)
    
    @pytest.mark.asyncio
    @patch.object(AliyunOCRModel, '_make_async_request')
    @patch.object(AliyunOCRModel, '_prepare_image_data')
    async def test_arecognize_text_success(self, mock_prepare, mock_request, ocr_model,
                                          sample_image_bytes, sample_image_base64, mock_aliyun_response):
        """测试成功的异步文字识别"""
        mock_prepare.return_value = sample_image_base64
        mock_request.return_value = mock_aliyun_response
        
        result = await ocr_model.arecognize_text(sample_image_bytes)
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "测试文本1\n测试文本2"
        mock_prepare.assert_called_once_with(sample_image_bytes)
        mock_request.assert_called_once_with(sample_image_base64)
    
    @patch.object(AliyunOCRModel, '_make_request')
    @patch.object(AliyunOCRModel, '_prepare_image_data')
    def test_recognize_text_with_retry(self, mock_prepare, mock_request, ocr_model,
                                      sample_image_bytes, sample_image_base64, mock_aliyun_response):
        """测试带重试机制的文字识别"""
        mock_prepare.return_value = sample_image_base64
        # 第一次调用失败，第二次成功
        mock_request.side_effect = [OCRError("临时错误"), mock_aliyun_response]
        
        result = ocr_model.recognize_text(sample_image_bytes)
        
        assert isinstance(result, OCRResult)
        assert mock_request.call_count == 2
    
    @patch.object(AliyunOCRModel, '_make_request')
    @patch.object(AliyunOCRModel, '_prepare_image_data')
    def test_recognize_text_max_retries_exceeded(self, mock_prepare, mock_request, ocr_model, sample_image_bytes):
        """测试超过最大重试次数"""
        mock_prepare.return_value = "fake_base64"
        mock_request.side_effect = OCRError("持续错误")
        
        with pytest.raises(OCRError, match="OCR识别失败，已重试"):
            ocr_model.recognize_text(sample_image_bytes)
        
        # 验证重试次数 = max_retries + 1 (初始尝试)
        assert mock_request.call_count == ocr_model.max_retries + 1
    
    def test_get_model_info(self, ocr_model):
        """测试获取模型信息"""
        info = ocr_model.get_model_info()
        
        assert info["model_name"] == "aliyun_ocr"
        assert info["type"] == "OCRModel"
        assert "max_retries" in info
        assert "timeout" in info


class TestAliyunOCRModelIntegration:
    """阿里云OCR模型集成测试"""
    
    def test_factory_registration(self):
        """测试工厂注册"""
        from core.models.ocr.ocr_factory import OCRFactory
        
        # 验证阿里云OCR已注册
        supported_providers = OCRFactory.get_supported_providers()
        assert "aliyun" in supported_providers
    
    def test_create_from_factory(self):
        """测试通过工厂创建实例"""
        from core.models.ocr.ocr_factory import OCRFactory
        
        with patch('core.models.ocr.cloud.aliyun_ocr_model.ALIYUN_OCR_API_KEY', 'test_key'):
            ocr = OCRFactory.create_ocr("aliyun", scene="table")
            assert isinstance(ocr, AliyunOCRModel)
            assert ocr.scene == "table"
    
    def test_import_from_main_module(self):
        """测试从主模块导入"""
        from core.models.ocr import AliyunOCRModel as ImportedAliyunOCRModel
        
        with patch('core.models.ocr.cloud.aliyun_ocr_model.ALIYUN_OCR_API_KEY', 'test_key'):
            ocr = ImportedAliyunOCRModel()
            assert isinstance(ocr, ImportedAliyunOCRModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
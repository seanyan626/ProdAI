# core/models/ocr/cloud/aliyun_ocr_model.py
"""
阿里云OCR模型实现

使用阿里云文字识别服务进行OCR识别。
支持通用文字识别、表格识别等多种场景。
"""

import base64
import json
import logging
import time
from typing import Union, List, Dict, Any, Optional
import requests
import asyncio
import aiohttp

from ..base_ocr_model import BaseOCRModel, OCRResult, OCRError
from configs.config import ALIYUN_OCR_API_KEY, OCR_TIMEOUT, OCR_MAX_RETRIES

logger = logging.getLogger(__name__)


class AliyunOCRModel(BaseOCRModel):
    """
    阿里云OCR模型实现
    
    使用阿里云文字识别服务API进行文字识别。
    支持多种识别场景，包括通用文字识别、表格识别等。
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "aliyun_ocr",
                 scene: str = "general",
                 **kwargs):
        """
        初始化阿里云OCR模型
        
        参数:
            api_key: 阿里云OCR API密钥，如果不提供则从配置中获取
            model_name: 模型名称
            scene: 识别场景，支持 'general'(通用), 'table'(表格), 'handwriting'(手写)等
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            max_retries=kwargs.get('max_retries', OCR_MAX_RETRIES),
            timeout=kwargs.get('timeout', OCR_TIMEOUT),
            **kwargs
        )
        
        self.api_key = api_key or ALIYUN_OCR_API_KEY
        self.scene = scene
        
        if not self.api_key:
            raise OCRError("阿里云OCR API密钥未配置。请在环境变量中设置 ALIYUN_OCR_API_KEY 或在初始化时提供 api_key 参数。")
        
        # 阿里云OCR API端点
        self.base_url = "https://ocr-api.cn-hangzhou.aliyuncs.com"
        
        # 根据场景设置API路径
        self.api_paths = {
            "general": "/api/predict/ocr_general",
            "table": "/api/predict/ocr_table_parse", 
            "handwriting": "/api/predict/ocr_handwriting"
        }
        
        if self.scene not in self.api_paths:
            logger.warning(f"不支持的识别场景: {self.scene}，将使用通用识别")
            self.scene = "general"
            
        self.api_url = self.base_url + self.api_paths[self.scene]
        
        logger.info(f"阿里云OCR模型已初始化，场景: {self.scene}")

    def _prepare_image_data(self, image_source: Union[str, bytes]) -> str:
        """
        准备图像数据，转换为base64编码
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            str: base64编码的图像数据
        """
        try:
            if isinstance(image_source, str):
                # 从文件路径读取图像
                with open(image_source, 'rb') as f:
                    image_bytes = f.read()
            elif isinstance(image_source, bytes):
                # 直接使用字节数据
                image_bytes = image_source
            else:
                raise OCRError(f"不支持的图像数据类型: {type(image_source)}")
            
            # 转换为base64编码
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return image_base64
            
        except FileNotFoundError:
            raise OCRError(f"图像文件未找到: {image_source}")
        except Exception as e:
            raise OCRError(f"准备图像数据失败: {str(e)}")

    def _parse_response(self, response_data: Dict[str, Any]) -> OCRResult:
        """
        解析阿里云OCR API响应
        
        参数:
            response_data: API响应数据
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            # 检查响应状态
            if response_data.get("success") != True:
                error_msg = response_data.get("message", "未知错误")
                raise OCRError(f"阿里云OCR API调用失败: {error_msg}")
            
            # 获取识别结果
            data = response_data.get("data", {})
            
            if self.scene == "general":
                return self._parse_general_response(data)
            elif self.scene == "table":
                return self._parse_table_response(data)
            elif self.scene == "handwriting":
                return self._parse_handwriting_response(data)
            else:
                return self._parse_general_response(data)
                
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"解析阿里云OCR响应失败: {str(e)}")

    def _parse_general_response(self, data: Dict[str, Any]) -> OCRResult:
        """解析通用文字识别响应"""
        full_text_parts = []
        segments = []
        total_confidence = 0.0
        
        # 获取文本行
        text_lines = data.get("content", [])
        
        for line in text_lines:
            text = line.get("text", "")
            confidence = line.get("prob", 0.0)
            
            # 获取边界框坐标
            text_rect = line.get("text_rect", [])
            bbox = []
            if text_rect and len(text_rect) >= 4:
                # 阿里云返回的是四个点的坐标，转换为 [x1, y1, x2, y2] 格式
                x_coords = [point[0] for point in text_rect]
                y_coords = [point[1] for point in text_rect]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            if text.strip():
                full_text_parts.append(text)
                segments.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
                total_confidence += confidence
        
        # 计算平均置信度
        avg_confidence = total_confidence / len(segments) if segments else 0.0
        full_text = "\n".join(full_text_parts)
        
        return OCRResult(
            full_text=full_text,
            segments=segments,
            confidence=avg_confidence
        )

    def _parse_table_response(self, data: Dict[str, Any]) -> OCRResult:
        """解析表格识别响应"""
        # 表格识别的响应格式可能不同，这里提供基础实现
        full_text_parts = []
        segments = []
        
        # 尝试从不同的字段获取文本内容
        if "tables" in data:
            for table in data["tables"]:
                for row in table.get("body", []):
                    for cell in row:
                        text = cell.get("text", "")
                        if text.strip():
                            full_text_parts.append(text)
                            segments.append({
                                "text": text,
                                "confidence": cell.get("confidence", 0.0),
                                "bbox": cell.get("bbox", [])
                            })
        
        full_text = "\n".join(full_text_parts)
        avg_confidence = sum(seg["confidence"] for seg in segments) / len(segments) if segments else 0.0
        
        return OCRResult(
            full_text=full_text,
            segments=segments,
            confidence=avg_confidence
        )

    def _parse_handwriting_response(self, data: Dict[str, Any]) -> OCRResult:
        """解析手写文字识别响应"""
        # 手写识别的响应格式，使用通用解析逻辑
        return self._parse_general_response(data)

    def _make_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送同步HTTP请求到阿里云OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        headers = {
            "Authorization": f"APPCODE {self.api_key}",
            "Content-Type": "application/json; charset=UTF-8"
        }
        
        payload = {
            "image": image_base64,
            "configure": {
                "min_size": 16,
                "output_prob": True,
                "output_keypoints": True
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise OCRError(f"阿里云OCR API请求超时 ({self.timeout}秒)")
        except requests.exceptions.HTTPError as e:
            raise OCRError(f"阿里云OCR API HTTP错误: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise OCRError(f"阿里云OCR API请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("阿里云OCR API返回的响应不是有效的JSON格式")

    async def _make_async_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送异步HTTP请求到阿里云OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        headers = {
            "Authorization": f"APPCODE {self.api_key}",
            "Content-Type": "application/json; charset=UTF-8"
        }
        
        payload = {
            "image": image_base64,
            "configure": {
                "min_size": 16,
                "output_prob": True,
                "output_keypoints": True
            }
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            raise OCRError(f"阿里云OCR API异步请求超时 ({self.timeout}秒)")
        except aiohttp.ClientResponseError as e:
            raise OCRError(f"阿里云OCR API HTTP错误: {e.status} - {e.message}")
        except aiohttp.ClientError as e:
            raise OCRError(f"阿里云OCR API异步请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("阿里云OCR API返回的响应不是有效的JSON格式")

    def recognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        同步识别图像中的文字
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            OCRResult: 识别结果
        """
        def _recognize():
            # 准备图像数据
            image_base64 = self._prepare_image_data(image_source)
            
            # 发送API请求
            response_data = self._make_request(image_base64)
            
            # 解析响应
            return self._parse_response(response_data)
        
        # 使用重试机制
        return self._retry_with_backoff(_recognize)

    async def arecognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        异步识别图像中的文字
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            OCRResult: 识别结果
        """
        async def _arecognize():
            # 准备图像数据
            image_base64 = self._prepare_image_data(image_source)
            
            # 发送异步API请求
            response_data = await self._make_async_request(image_base64)
            
            # 解析响应
            return self._parse_response(response_data)
        
        # 使用异步重试机制
        return await self._async_retry_with_backoff(_arecognize)


if __name__ == "__main__":
    # 测试代码
    from configs.config import load_config
    from configs.logging_config import setup_logging
    
    load_config()
    setup_logging()
    
    logger.info("阿里云OCR模型测试")
    
    try:
        # 创建阿里云OCR模型实例
        ocr = AliyunOCRModel(scene="general")
        logger.info(f"模型信息: {ocr.get_model_info()}")
        
        # 注意：实际测试需要有效的API密钥和图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")
        
    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
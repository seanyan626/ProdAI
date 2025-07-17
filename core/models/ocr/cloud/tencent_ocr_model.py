# core/models/ocr/cloud/tencent_ocr_model.py
"""
腾讯云OCR模型实现

使用腾讯云文字识别服务进行OCR识别。
支持通用文字识别、表格识别等多种场景。
"""

import base64
import json
import logging
import time
import hmac
import hashlib
from typing import Union, List, Dict, Any, Optional
import requests
import asyncio
import aiohttp
from datetime import datetime

from ..base_ocr_model import BaseOCRModel, OCRResult, OCRError
from configs.config import OCR_TIMEOUT, OCR_MAX_RETRIES

logger = logging.getLogger(__name__)

# 腾讯云OCR配置变量
TENCENT_OCR_SECRET_ID = None
TENCENT_OCR_SECRET_KEY = None


class TencentOCRModel(BaseOCRModel):
    """
    腾讯云OCR模型实现
    
    使用腾讯云文字识别服务API进行文字识别。
    支持多种识别场景，包括通用文字识别、表格识别等。
    """
    
    def __init__(self, 
                 secret_id: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 model_name: str = "tencent_ocr",
                 scene: str = "general",
                 **kwargs):
        """
        初始化腾讯云OCR模型
        
        参数:
            secret_id: 腾讯云API密钥ID，如果不提供则从配置中获取
            secret_key: 腾讯云API密钥，如果不提供则从配置中获取
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
        
        # 从全局配置或参数中获取密钥
        global TENCENT_OCR_SECRET_ID, TENCENT_OCR_SECRET_KEY
        self.secret_id = secret_id or TENCENT_OCR_SECRET_ID
        self.secret_key = secret_key or TENCENT_OCR_SECRET_KEY
        
        if not self.secret_id or not self.secret_key:
            raise OCRError("腾讯云OCR API密钥未配置。请在环境变量中设置 TENCENT_OCR_SECRET_ID 和 TENCENT_OCR_SECRET_KEY 或在初始化时提供 secret_id 和 secret_key 参数。")
        
        self.scene = scene
        
        # 腾讯云OCR API端点和服务信息
        self.host = "ocr.tencentcloudapi.com"
        self.service = "ocr"
        self.region = "ap-guangzhou"  # 默认区域
        self.version = "2018-11-19"   # API版本
        
        # 根据场景设置API操作
        self.api_actions = {
            "general": "GeneralBasicOCR",        # 通用印刷体识别
            "accurate": "GeneralAccurateOCR",    # 通用印刷体识别（高精度版）
            "handwriting": "GeneralHandwritingOCR",  # 通用手写体识别
            "table": "TableOCR",                 # 表格识别
            "id_card": "IDCardOCR",              # 身份证识别
            "business_card": "BusinessCardOCR",  # 名片识别
            "license": "LicenseOCR",             # 营业执照识别
            "invoice": "InvoiceGeneralOCR"       # 通用发票识别
        }
        
        if self.scene not in self.api_actions:
            logger.warning(f"不支持的识别场景: {self.scene}，将使用通用识别")
            self.scene = "general"
            
        self.action = self.api_actions[self.scene]
        
        logger.info(f"腾讯云OCR模型已初始化，场景: {self.scene}, 操作: {self.action}")

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

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        生成腾讯云API请求签名
        
        参数:
            params: 请求参数
            
        返回:
            str: 签名字符串
        """
        try:
            # 1. 按字典序排序参数
            sorted_params = sorted(params.items(), key=lambda x: x[0])
            
            # 2. 拼接请求字符串
            request_str = "POST" + self.host + "/?" + "&".join(f"{k}={v}" for k, v in sorted_params)
            
            # 3. 使用HMAC-SHA1算法计算签名
            hmac_str = hmac.new(
                self.secret_key.encode('utf-8'),
                request_str.encode('utf-8'),
                hashlib.sha1
            ).digest()
            
            # 4. Base64编码
            signature = base64.b64encode(hmac_str).decode('utf-8')
            
            return signature
            
        except Exception as e:
            raise OCRError(f"生成腾讯云API签名失败: {str(e)}")

    def _get_common_params(self) -> Dict[str, str]:
        """
        获取腾讯云API公共参数
        
        返回:
            Dict: 公共参数字典
        """
        # 获取当前时间戳
        timestamp = int(time.time())
        
        # 生成随机数
        nonce = int(time.time() * 1000) % 65535
        
        return {
            "Action": self.action,
            "Region": self.region,
            "Timestamp": str(timestamp),
            "Nonce": str(nonce),
            "SecretId": self.secret_id,
            "Version": self.version,
        }

    def _parse_response(self, response_data: Dict[str, Any]) -> OCRResult:
        """
        解析腾讯云OCR API响应
        
        参数:
            response_data: API响应数据
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            # 检查响应是否包含错误
            if "Response" not in response_data:
                error_msg = response_data.get("error", {}).get("message", "未知错误")
                raise OCRError(f"腾讯云OCR API调用失败: {error_msg}")
            
            response = response_data["Response"]
            
            # 检查是否有错误
            if "Error" in response:
                error = response["Error"]
                error_code = error.get("Code", "UnknownError")
                error_msg = error.get("Message", "未知错误")
                raise OCRError(f"腾讯云OCR API错误: [{error_code}] {error_msg}")
            
            # 根据不同场景解析响应
            if self.scene == "general" or self.scene == "accurate":
                return self._parse_general_response(response)
            elif self.scene == "table":
                return self._parse_table_response(response)
            elif self.scene == "handwriting":
                return self._parse_handwriting_response(response)
            else:
                # 默认使用通用解析
                return self._parse_general_response(response)
                
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"解析腾讯云OCR响应失败: {str(e)}")

    def _parse_general_response(self, response: Dict[str, Any]) -> OCRResult:
        """解析通用文字识别响应"""
        full_text_parts = []
        segments = []
        total_confidence = 0.0
        
        # 获取文本行
        text_detections = response.get("TextDetections", [])
        
        for detection in text_detections:
            text = detection.get("DetectedText", "")
            confidence = detection.get("Confidence", 0.0) / 100.0  # 腾讯云返回的置信度是0-100，转换为0-1
            
            # 获取边界框坐标
            text_coords = detection.get("Polygon", [])
            bbox = []
            if text_coords and len(text_coords) >= 4:
                x_coords = [point.get("X", 0) for point in text_coords]
                y_coords = [point.get("Y", 0) for point in text_coords]
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

    def _parse_table_response(self, response: Dict[str, Any]) -> OCRResult:
        """解析表格识别响应"""
        full_text_parts = []
        segments = []
        
        # 表格数据
        table_detections = response.get("TableDetections", [])
        
        for table in table_detections:
            cells = table.get("Cells", [])
            for cell in cells:
                text = cell.get("Text", "")
                row = cell.get("Row", 0)
                col = cell.get("Col", 0)
                
                if text.strip():
                    full_text_parts.append(text)
                    segments.append({
                        "text": text,
                        "row": row,
                        "col": col,
                        "confidence": 0.9,  # 腾讯云表格识别API可能不返回置信度，使用默认值
                        "bbox": []  # 腾讯云表格识别API可能不返回边界框
                    })
        
        full_text = "\n".join(full_text_parts)
        avg_confidence = 0.9  # 默认置信度
        
        return OCRResult(
            full_text=full_text,
            segments=segments,
            confidence=avg_confidence
        )

    def _parse_handwriting_response(self, response: Dict[str, Any]) -> OCRResult:
        """解析手写文字识别响应"""
        # 手写识别的响应格式与通用识别类似
        return self._parse_general_response(response)

    def _make_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送同步HTTP请求到腾讯云OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        # 准备请求参数
        params = self._get_common_params()
        
        # 添加业务参数
        params["ImageBase64"] = image_base64
        
        # 生成签名
        signature = self._generate_signature(params)
        params["Signature"] = signature
        
        # 构建请求URL
        url = f"https://{self.host}"
        
        try:
            response = requests.post(
                url,
                data=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise OCRError(f"腾讯云OCR API请求超时 ({self.timeout}秒)")
        except requests.exceptions.HTTPError as e:
            raise OCRError(f"腾讯云OCR API HTTP错误: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise OCRError(f"腾讯云OCR API请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("腾讯云OCR API返回的响应不是有效的JSON格式")

    async def _make_async_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送异步HTTP请求到腾讯云OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        # 准备请求参数
        params = self._get_common_params()
        
        # 添加业务参数
        params["ImageBase64"] = image_base64
        
        # 生成签名
        signature = self._generate_signature(params)
        params["Signature"] = signature
        
        # 构建请求URL
        url = f"https://{self.host}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    data=params
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            raise OCRError(f"腾讯云OCR API异步请求超时 ({self.timeout}秒)")
        except aiohttp.ClientResponseError as e:
            raise OCRError(f"腾讯云OCR API HTTP错误: {e.status} - {e.message}")
        except aiohttp.ClientError as e:
            raise OCRError(f"腾讯云OCR API异步请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("腾讯云OCR API返回的响应不是有效的JSON格式")

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
    
    logger.info("腾讯云OCR模型测试")
    
    try:
        # 创建腾讯云OCR模型实例
        ocr = TencentOCRModel(scene="general")
        logger.info(f"模型信息: {ocr.get_model_info()}")
        
        # 注意：实际测试需要有效的API密钥和图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")
        
    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
# core/models/ocr/cloud/baidu_ocr_model.py
"""
百度OCR模型实现

使用百度智能云文字识别服务进行OCR识别。
支持通用文字识别、高精度文字识别、表格识别等多种场景。
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
from configs.config import BAIDU_OCR_API_KEY, BAIDU_OCR_SECRET_KEY, OCR_TIMEOUT, OCR_MAX_RETRIES

logger = logging.getLogger(__name__)


class BaiduOCRModel(BaseOCRModel):
    """
    百度OCR模型实现
    
    使用百度智能云文字识别服务API进行文字识别。
    支持多种识别场景，包括通用文字识别、高精度识别、表格识别等。
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 model_name: str = "baidu_ocr",
                 scene: str = "general_basic",
                 **kwargs):
        """
        初始化百度OCR模型
        
        参数:
            api_key: 百度OCR API Key，如果不提供则从配置中获取
            secret_key: 百度OCR Secret Key，如果不提供则从配置中获取
            model_name: 模型名称
            scene: 识别场景，支持 'general_basic'(通用基础), 'accurate_basic'(高精度基础), 
                   'general'(通用含位置), 'accurate'(高精度含位置), 'table'(表格)等
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            max_retries=kwargs.get('max_retries', OCR_MAX_RETRIES),
            timeout=kwargs.get('timeout', OCR_TIMEOUT),
            **kwargs
        )
        
        self.api_key = api_key or BAIDU_OCR_API_KEY
        self.secret_key = secret_key or BAIDU_OCR_SECRET_KEY
        self.scene = scene
        
        if not self.api_key or not self.secret_key:
            raise OCRError("百度OCR API密钥未配置。请在环境变量中设置 BAIDU_OCR_API_KEY 和 BAIDU_OCR_SECRET_KEY 或在初始化时提供相应参数。")
        
        # 百度OCR API端点
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.base_url = "https://aip.baidubce.com/rest/2.0/ocr/v1"
        
        # 根据场景设置API路径
        self.api_paths = {
            "general_basic": "/general_basic",
            "accurate_basic": "/accurate_basic", 
            "general": "/general",
            "accurate": "/accurate",
            "table": "/table"
        }
        
        if self.scene not in self.api_paths:
            logger.warning(f"不支持的识别场景: {self.scene}，将使用通用基础识别")
            self.scene = "general_basic"
            
        self.api_url = self.base_url + self.api_paths[self.scene]
        
        # 访问令牌缓存
        self._access_token = None
        self._token_expires_at = 0
        
        logger.info(f"百度OCR模型已初始化，场景: {self.scene}")

    def _get_access_token(self) -> str:
        """
        获取百度API访问令牌
        
        返回:
            str: 访问令牌
        """
        # 检查令牌是否仍然有效（提前5分钟刷新）
        if self._access_token and time.time() < (self._token_expires_at - 300):
            return self._access_token
        
        try:
            params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }
            
            response = requests.post(
                self.token_url,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            if "access_token" not in token_data:
                error_msg = token_data.get("error_description", "获取访问令牌失败")
                raise OCRError(f"百度OCR获取访问令牌失败: {error_msg}")
            
            self._access_token = token_data["access_token"]
            # 令牌有效期通常为30天，这里设置为当前时间 + 返回的有效期（秒）
            expires_in = token_data.get("expires_in", 2592000)  # 默认30天
            self._token_expires_at = time.time() + expires_in
            
            logger.info("百度OCR访问令牌已更新")
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            raise OCRError(f"获取百度OCR访问令牌失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("百度OCR令牌API返回的响应不是有效的JSON格式")

    async def _get_access_token_async(self) -> str:
        """
        异步获取百度API访问令牌
        
        返回:
            str: 访问令牌
        """
        # 检查令牌是否仍然有效（提前5分钟刷新）
        if self._access_token and time.time() < (self._token_expires_at - 300):
            return self._access_token
        
        try:
            params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.token_url, params=params) as response:
                    response.raise_for_status()
                    token_data = await response.json()
            
            if "access_token" not in token_data:
                error_msg = token_data.get("error_description", "获取访问令牌失败")
                raise OCRError(f"百度OCR获取访问令牌失败: {error_msg}")
            
            self._access_token = token_data["access_token"]
            # 令牌有效期通常为30天
            expires_in = token_data.get("expires_in", 2592000)  # 默认30天
            self._token_expires_at = time.time() + expires_in
            
            logger.info("百度OCR访问令牌已更新（异步）")
            return self._access_token
            
        except aiohttp.ClientError as e:
            raise OCRError(f"异步获取百度OCR访问令牌失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("百度OCR令牌API返回的响应不是有效的JSON格式")

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
        解析百度OCR API响应
        
        参数:
            response_data: API响应数据
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            # 检查响应中是否有错误
            if "error_code" in response_data:
                error_code = response_data["error_code"]
                error_msg = response_data.get("error_msg", f"错误代码: {error_code}")
                raise OCRError(f"百度OCR API调用失败: {error_msg}")
            
            # 获取识别结果
            words_result = response_data.get("words_result", [])
            
            if self.scene == "table":
                return self._parse_table_response(response_data)
            else:
                return self._parse_text_response(words_result)
                
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"解析百度OCR响应失败: {str(e)}")

    def _parse_text_response(self, words_result: List[Dict[str, Any]]) -> OCRResult:
        """解析文字识别响应"""
        full_text_parts = []
        segments = []
        total_confidence = 0.0
        
        for word_info in words_result:
            text = word_info.get("words", "")
            
            # 获取置信度（如果可用）
            confidence = 1.0  # 基础版本不返回置信度，设为1.0
            if "probability" in word_info:
                # 高精度版本可能包含置信度信息
                prob_info = word_info["probability"]
                if isinstance(prob_info, dict) and "average" in prob_info:
                    confidence = prob_info["average"]
                elif isinstance(prob_info, (int, float)):
                    confidence = float(prob_info)
            
            # 获取位置信息（如果可用）
            bbox = []
            if "location" in word_info:
                location = word_info["location"]
                # 百度返回的位置格式: {"left": x, "top": y, "width": w, "height": h}
                left = location.get("left", 0)
                top = location.get("top", 0)
                width = location.get("width", 0)
                height = location.get("height", 0)
                bbox = [left, top, left + width, top + height]
            
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

    def _parse_table_response(self, response_data: Dict[str, Any]) -> OCRResult:
        """解析表格识别响应"""
        full_text_parts = []
        segments = []
        
        # 表格识别的响应格式
        forms = response_data.get("forms", [])
        
        for form in forms:
            body = form.get("body", [])
            for row in body:
                for cell in row:
                    text = cell.get("word", "")
                    if text.strip():
                        full_text_parts.append(text)
                        
                        # 获取位置信息
                        bbox = []
                        if "location" in cell:
                            location = cell["location"]
                            left = location.get("left", 0)
                            top = location.get("top", 0)
                            width = location.get("width", 0)
                            height = location.get("height", 0)
                            bbox = [left, top, left + width, top + height]
                        
                        segments.append({
                            "text": text,
                            "confidence": 1.0,  # 表格识别通常不返回置信度
                            "bbox": bbox
                        })
        
        full_text = "\n".join(full_text_parts)
        avg_confidence = 1.0
        
        return OCRResult(
            full_text=full_text,
            segments=segments,
            confidence=avg_confidence
        )

    def _make_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送同步HTTP请求到百度OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        # 获取访问令牌
        access_token = self._get_access_token()
        
        # 构建请求URL
        url = f"{self.api_url}?access_token={access_token}"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # 构建请求数据
        data = {
            "image": image_base64
        }
        
        # 根据场景添加额外参数
        if self.scene in ["general", "accurate"]:
            data["recognize_granularity"] = "big"  # 识别粒度
            data["probability"] = "true"  # 返回置信度
        
        try:
            response = requests.post(
                url,
                headers=headers,
                data=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise OCRError(f"百度OCR API请求超时 ({self.timeout}秒)")
        except requests.exceptions.HTTPError as e:
            raise OCRError(f"百度OCR API HTTP错误: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise OCRError(f"百度OCR API请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("百度OCR API返回的响应不是有效的JSON格式")

    async def _make_async_request(self, image_base64: str) -> Dict[str, Any]:
        """
        发送异步HTTP请求到百度OCR API
        
        参数:
            image_base64: base64编码的图像数据
            
        返回:
            Dict: API响应数据
        """
        # 获取访问令牌
        access_token = await self._get_access_token_async()
        
        # 构建请求URL
        url = f"{self.api_url}?access_token={access_token}"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # 构建请求数据
        data = {
            "image": image_base64
        }
        
        # 根据场景添加额外参数
        if self.scene in ["general", "accurate"]:
            data["recognize_granularity"] = "big"  # 识别粒度
            data["probability"] = "true"  # 返回置信度
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=data) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except asyncio.TimeoutError:
            raise OCRError(f"百度OCR API异步请求超时 ({self.timeout}秒)")
        except aiohttp.ClientResponseError as e:
            raise OCRError(f"百度OCR API HTTP错误: {e.status} - {e.message}")
        except aiohttp.ClientError as e:
            raise OCRError(f"百度OCR API异步请求失败: {str(e)}")
        except json.JSONDecodeError:
            raise OCRError("百度OCR API返回的响应不是有效的JSON格式")

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
    
    logger.info("百度OCR模型测试")
    
    try:
        # 创建百度OCR模型实例
        ocr = BaiduOCRModel(scene="general_basic")
        logger.info(f"模型信息: {ocr.get_model_info()}")
        
        # 注意：实际测试需要有效的API密钥和图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")
        
    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
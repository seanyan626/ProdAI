# core/models/ocr/local/easy_ocr_model.py
"""
EasyOCR本地模型实现

使用EasyOCR库进行本地文字识别。
支持多语言识别和自动语言检测。
"""

import logging
import os
import asyncio
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import io

from ..base_ocr_model import BaseOCRModel, OCRResult, OCRError
from configs.config import OCR_TIMEOUT, OCR_MAX_RETRIES

logger = logging.getLogger(__name__)

# 延迟导入EasyOCR，避免在导入模块时就加载模型
try:
    import easyocr
    import torch
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR库未安装，请使用 'pip install easyocr' 安装")


class EasyOCRModel(BaseOCRModel):
    """
    EasyOCR本地模型实现
    
    使用EasyOCR库进行本地文字识别。
    支持多语言识别和自动语言检测。
    """
    
    def __init__(self, 
                 languages: Optional[List[str]] = None,
                 model_name: str = "easyocr",
                 gpu: bool = True,
                 model_storage_directory: Optional[str] = None,
                 download_enabled: bool = True,
                 **kwargs):
        """
        初始化EasyOCR模型
        
        参数:
            languages: 要识别的语言列表，如 ['ch_sim', 'en']。如果为None，则使用英文
            model_name: 模型名称
            gpu: 是否使用GPU加速
            model_storage_directory: 模型存储目录，如果为None则使用默认目录
            download_enabled: 是否允许自动下载模型
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            max_retries=kwargs.get('max_retries', OCR_MAX_RETRIES),
            timeout=kwargs.get('timeout', OCR_TIMEOUT),
            **kwargs
        )
        
        if not EASYOCR_AVAILABLE:
            raise OCRError("EasyOCR库未安装，请使用 'pip install easyocr' 安装")
        
        # 默认使用英文
        self.languages = languages or ['en']
        
        # 检查GPU可用性
        self.gpu = gpu and torch.cuda.is_available()
        if gpu and not torch.cuda.is_available():
            logger.warning("GPU不可用，将使用CPU模式")
        
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        
        # 延迟初始化Reader，直到第一次调用
        self._reader = None
        
        logger.info(f"EasyOCR模型已初始化，语言: {self.languages}, GPU: {self.gpu}")

    def _get_reader(self):
        """
        获取或初始化EasyOCR Reader
        
        返回:
            easyocr.Reader: EasyOCR Reader实例
        """
        if self._reader is None:
            try:
                logger.info(f"初始化EasyOCR Reader，语言: {self.languages}, GPU: {self.gpu}")
                
                # 如果不允许下载，检查模型文件是否存在
                if not self.download_enabled:
                    self._check_model_files()
                
                # 初始化Reader
                self._reader = easyocr.Reader(
                    lang_list=self.languages,
                    gpu=self.gpu,
                    model_storage_directory=self.model_storage_directory,
                    download_enabled=self.download_enabled,
                    # 可选参数
                    quantize=self.config.get('quantize', False),  # 模型量化，减少内存使用
                    verbose=self.config.get('verbose', False)     # 是否显示详细日志
                )
                
                logger.info("EasyOCR Reader初始化完成")
            except Exception as e:
                raise OCRError(f"初始化EasyOCR Reader失败: {str(e)}")
        
        return self._reader

    def _check_model_files(self):
        """
        检查模型文件是否存在，如果不存在且不允许下载则抛出异常
        """
        if not self.model_storage_directory:
            # 使用默认目录
            home_dir = os.path.expanduser("~")
            self.model_storage_directory = os.path.join(home_dir, '.EasyOCR', 'model')
        
        # 检查模型文件
        for lang in self.languages:
            # 检查detection模型
            detection_model = os.path.join(self.model_storage_directory, 'craft_mlt_25k.pth')
            if not os.path.exists(detection_model):
                raise OCRError(f"检测模型文件不存在: {detection_model}，且不允许自动下载")
            
            # 检查recognition模型
            if lang != 'en':
                recognition_model = os.path.join(self.model_storage_directory, f"{lang}_g2.pth")
                if not os.path.exists(recognition_model):
                    raise OCRError(f"识别模型文件不存在: {recognition_model}，且不允许自动下载")

    def _prepare_image(self, image_source: Union[str, bytes]) -> Union[str, np.ndarray]:
        """
        准备图像数据
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            Union[str, np.ndarray]: 图像路径或numpy数组
        """
        try:
            if isinstance(image_source, str):
                # 如果是文件路径，直接返回
                if not os.path.exists(image_source):
                    raise OCRError(f"图像文件不存在: {image_source}")
                return image_source
            
            elif isinstance(image_source, bytes):
                # 如果是字节数据，转换为numpy数组
                try:
                    image = Image.open(io.BytesIO(image_source))
                    return np.array(image)
                except Exception as e:
                    raise OCRError(f"无法从字节数据加载图像: {str(e)}")
            
            else:
                raise OCRError(f"不支持的图像数据类型: {type(image_source)}")
                
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"准备图像数据失败: {str(e)}")

    def _parse_result(self, result: List[List[Any]]) -> OCRResult:
        """
        解析EasyOCR识别结果
        
        参数:
            result: EasyOCR识别结果
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            full_text_parts = []
            segments = []
            total_confidence = 0.0
            
            for detection in result:
                # EasyOCR结果格式: [bbox, text, confidence]
                # bbox格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                bbox, text, confidence = detection
                
                if text.strip():
                    full_text_parts.append(text)
                    
                    # 转换bbox为[x1, y1, x2, y2]格式
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    segments.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox_rect
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
            
        except Exception as e:
            raise OCRError(f"解析EasyOCR结果失败: {str(e)}")

    def _detect_language(self, image_source: Union[str, bytes]) -> List[str]:
        """
        检测图像中的语言
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            List[str]: 检测到的语言代码列表
        """
        # 注意：EasyOCR本身不提供语言检测功能
        # 这里使用一个简单的方法：先用英文模型识别，如果置信度低，则尝试其他语言
        
        # 如果已经指定了语言，直接返回
        if self.languages != ['en']:
            return self.languages
        
        try:
            # 创建一个临时的英文Reader
            temp_reader = easyocr.Reader(
                lang_list=['en'],
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                verbose=False
            )
            
            # 准备图像
            image = self._prepare_image(image_source)
            
            # 识别文本
            result = temp_reader.readtext(image, detail=1)
            
            # 计算平均置信度
            if not result:
                # 没有识别到文本，默认返回英文和中文简体
                return ['en', 'ch_sim']
            
            avg_confidence = sum(item[2] for item in result) / len(result)
            
            # 如果英文置信度低，可能是其他语言
            if avg_confidence < 0.5:
                # 返回常用语言组合
                return ['en', 'ch_sim']
            
            return ['en']
            
        except Exception as e:
            logger.warning(f"语言检测失败: {str(e)}，将使用默认语言: {self.languages}")
            return self.languages

    def recognize_text(self, image_source: Union[str, bytes]) -> OCRResult:
        """
        识别图像中的文字
        
        参数:
            image_source: 图像路径或字节数据
            
        返回:
            OCRResult: 识别结果
        """
        def _recognize():
            # 准备图像
            image = self._prepare_image(image_source)
            
            # 如果启用了自动语言检测且未指定语言
            if self.config.get('auto_language_detection', False) and self.languages == ['en']:
                detected_languages = self._detect_language(image_source)
                
                # 如果检测到的语言与当前不同，重新初始化Reader
                if set(detected_languages) != set(self.languages):
                    logger.info(f"检测到语言: {detected_languages}，重新初始化Reader")
                    self.languages = detected_languages
                    self._reader = None
            
            # 获取Reader
            reader = self._get_reader()
            
            # 识别文本
            # detail=1表示返回详细信息，包括边界框和置信度
            # paragraph=True表示尝试将相邻文本组合成段落
            result = reader.readtext(
                image,
                detail=1,
                paragraph=self.config.get('paragraph', False),
                batch_size=self.config.get('batch_size', 1),
                min_size=self.config.get('min_size', 20),
                contrast_ths=self.config.get('contrast_ths', 0.1),
                adjust_contrast=self.config.get('adjust_contrast', 0.5),
                text_threshold=self.config.get('text_threshold', 0.7),
                link_threshold=self.config.get('link_threshold', 0.4),
                low_text=self.config.get('low_text', 0.4),
                canvas_size=self.config.get('canvas_size', 2560),
                mag_ratio=self.config.get('mag_ratio', 1.0),
                slope_ths=self.config.get('slope_ths', 0.1),
                ycenter_ths=self.config.get('ycenter_ths', 0.5),
                height_ths=self.config.get('height_ths', 0.5),
                width_ths=self.config.get('width_ths', 0.5),
                y_ths=self.config.get('y_ths', 0.5),
                x_ths=self.config.get('x_ths', 1.0),
                add_margin=self.config.get('add_margin', 0.1),
            )
            
            # 解析结果
            return self._parse_result(result)
        
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
        # 由于EasyOCR不是异步库，我们使用线程池来异步执行
        loop = asyncio.get_event_loop()
        
        async def _arecognize():
            # 在线程池中执行同步方法
            return await loop.run_in_executor(None, self.recognize_text, image_source)
        
        # 使用异步重试机制
        return await self._async_retry_with_backoff(_arecognize)

    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取EasyOCR支持的语言列表
        
        返回:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        # EasyOCR支持的语言
        return {
            'en': 'English',
            'ch_sim': 'Chinese (Simplified)',
            'ch_tra': 'Chinese (Traditional)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'th': 'Thai',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'hi': 'Hindi',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'bn': 'Bengali',
            'ar': 'Arabic',
            'fa': 'Persian',
            'ur': 'Urdu',
            'sr': 'Serbian',
            'bg': 'Bulgarian',
            'ru': 'Russian',
            'rs_cyrillic': 'Russian/Serbian (Cyrillic)',
            'be': 'Belarusian',
            'uk': 'Ukrainian',
            'mn': 'Mongolian',
            'am': 'Amharic',
            'fr': 'French',
            'it': 'Italian',
            'de': 'German',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'cs': 'Czech',
            'pl': 'Polish',
            'ro': 'Romanian',
            'lv': 'Latvian',
            'hu': 'Hungarian',
            'el': 'Greek',
            'cy': 'Welsh',
            'hr': 'Croatian',
            'he': 'Hebrew',
            'gl': 'Galician',
            'da': 'Danish',
            'sv': 'Swedish',
            'sk': 'Slovak',
            'id': 'Indonesian',
            'ms': 'Malay',
            'ca': 'Catalan',
            'bs': 'Bosnian',
            'sl': 'Slovenian',
            'fi': 'Finnish',
            'no': 'Norwegian',
            'eo': 'Esperanto',
            'az': 'Azerbaijani',
            'uz': 'Uzbek',
            'kk': 'Kazakh',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'si': 'Sinhala',
            'dv': 'Dhivehi',
            'af': 'Afrikaans',
            'et': 'Estonian',
            'is': 'Icelandic',
            'ht': 'Haitian',
            'ku': 'Kurdish',
            'ky': 'Kyrgyz',
            'mi': 'Maori',
            'mg': 'Malagasy',
            'tl': 'Tagalog',
            'tt': 'Tatar',
            'tk': 'Turkmen',
            'ug': 'Uyghur',
            'yi': 'Yiddish',
        }

    def get_model_info(self) -> dict:
        """
        返回关于OCR模型的信息
        
        返回:
            dict: 模型信息
        """
        info = super().get_model_info()
        info.update({
            "languages": self.languages,
            "gpu": self.gpu,
            "model_storage_directory": self.model_storage_directory,
            "download_enabled": self.download_enabled,
            "supported_languages": list(self.get_supported_languages().keys()),
            "type": "LocalOCRModel"
        })
        return info


if __name__ == "__main__":
    # 测试代码
    from configs.config import load_config
    from configs.logging_config import setup_logging
    
    load_config()
    setup_logging()
    
    logger.info("EasyOCR模型测试")
    
    try:
        # 创建EasyOCR模型实例
        ocr = EasyOCRModel(languages=['en', 'ch_sim'], gpu=False)
        logger.info(f"模型信息: {ocr.get_model_info()}")
        
        # 注意：实际测试需要有效的图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")
        
    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
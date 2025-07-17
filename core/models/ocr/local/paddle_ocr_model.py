# core/models/ocr/local/paddle_ocr_model.py
"""
PaddleOCR本地模型实现

使用PaddleOCR库进行本地文字识别。
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

# 延迟导入PaddleOCR，避免在导入模块时就加载模型
try:
    from paddleocr import PaddleOCR, paddleocr
    import paddle
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR库未安装，请使用 'pip install paddleocr' 安装")


class PaddleOCRModel(BaseOCRModel):
    """
    PaddleOCR本地模型实现
    
    使用PaddleOCR库进行本地文字识别。
    支持多语言识别和自动语言检测。
    """
    
    def __init__(self, 
                 lang: str = "ch",
                 model_name: str = "paddleocr",
                 use_gpu: bool = True,
                 use_angle_cls: bool = True,
                 det: bool = True,
                 rec: bool = True,
                 cls: bool = True,
                 enable_mkldnn: bool = False,
                 cpu_threads: int = 10,
                 det_model_dir: Optional[str] = None,
                 rec_model_dir: Optional[str] = None,
                 cls_model_dir: Optional[str] = None,
                 download_enabled: bool = True,
                 **kwargs):
        """
        初始化PaddleOCR模型
        
        参数:
            lang: 要识别的语言，如 'ch', 'en', 'fr', 'german', 'korean', 'japan'
            model_name: 模型名称
            use_gpu: 是否使用GPU加速
            use_angle_cls: 是否使用方向分类器
            det: 是否进行文本检测
            rec: 是否进行文本识别
            cls: 是否使用方向分类器
            enable_mkldnn: 是否启用mkldnn加速
            cpu_threads: CPU线程数
            det_model_dir: 检测模型目录，如果为None则使用默认目录
            rec_model_dir: 识别模型目录，如果为None则使用默认目录
            cls_model_dir: 方向分类模型目录，如果为None则使用默认目录
            download_enabled: 是否允许自动下载模型
            **kwargs: 其他配置参数
        """
        super().__init__(
            model_name=model_name,
            max_retries=kwargs.get('max_retries', OCR_MAX_RETRIES),
            timeout=kwargs.get('timeout', OCR_TIMEOUT),
            **kwargs
        )
        
        if not PADDLEOCR_AVAILABLE:
            raise OCRError("PaddleOCR库未安装，请使用 'pip install paddleocr' 安装")
        
        # 检查GPU可用性
        self.use_gpu = use_gpu and paddle.device.is_compiled_with_cuda()
        if use_gpu and not paddle.device.is_compiled_with_cuda():
            logger.warning("GPU不可用，将使用CPU模式")
        
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det = det
        self.rec = rec
        self.cls = cls
        self.enable_mkldnn = enable_mkldnn
        self.cpu_threads = cpu_threads
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.download_enabled = download_enabled
        
        # 延迟初始化PaddleOCR，直到第一次调用
        self._ocr = None
        
        logger.info(f"PaddleOCR模型已初始化，语言: {self.lang}, GPU: {self.use_gpu}")

    def _get_ocr(self):
        """
        获取或初始化PaddleOCR实例
        
        返回:
            PaddleOCR: PaddleOCR实例
        """
        if self._ocr is None:
            try:
                logger.info(f"初始化PaddleOCR，语言: {self.lang}, GPU: {self.use_gpu}")
                
                # 如果不允许下载，检查模型文件是否存在
                if not self.download_enabled:
                    self._check_model_files()
                
                # 初始化PaddleOCR
                self._ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    det=self.det,
                    rec=self.rec,
                    cls=self.cls,
                    enable_mkldnn=self.enable_mkldnn,
                    cpu_threads=self.cpu_threads,
                    det_model_dir=self.det_model_dir,
                    rec_model_dir=self.rec_model_dir,
                    cls_model_dir=self.cls_model_dir,
                    # 可选参数
                    show_log=self.config.get('show_log', False),
                    use_mp=self.config.get('use_mp', False),
                    total_process_num=self.config.get('total_process_num', 1),
                    precision=self.config.get('precision', 'fp32'),
                )
                
                logger.info("PaddleOCR初始化完成")
            except Exception as e:
                raise OCRError(f"初始化PaddleOCR失败: {str(e)}")
        
        return self._ocr

    def _check_model_files(self):
        """
        检查模型文件是否存在，如果不存在且不允许下载则抛出异常
        """
        # 获取默认模型目录
        home_dir = os.path.expanduser("~")
        paddle_home = os.path.join(home_dir, '.paddleocr')
        
        # 检测模型
        if not self.det_model_dir:
            det_model_dir = os.path.join(paddle_home, 'whl', 'det', f'ch_PP-OCRv3_det_infer')
            if not os.path.exists(det_model_dir):
                raise OCRError(f"检测模型目录不存在: {det_model_dir}，且不允许自动下载")
        
        # 识别模型
        if not self.rec_model_dir:
            rec_model_dir = os.path.join(paddle_home, 'whl', 'rec', f'{self.lang}_PP-OCRv3_rec_infer')
            if not os.path.exists(rec_model_dir):
                raise OCRError(f"识别模型目录不存在: {rec_model_dir}，且不允许自动下载")
        
        # 方向分类模型
        if self.use_angle_cls and not self.cls_model_dir:
            cls_model_dir = os.path.join(paddle_home, 'whl', 'cls', f'ch_ppocr_mobile_v2.0_cls_infer')
            if not os.path.exists(cls_model_dir):
                raise OCRError(f"方向分类模型目录不存在: {cls_model_dir}，且不允许自动下载")

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

    def _parse_result(self, result: List) -> OCRResult:
        """
        解析PaddleOCR识别结果
        
        参数:
            result: PaddleOCR识别结果
            
        返回:
            OCRResult: 解析后的OCR结果
        """
        try:
            full_text_parts = []
            segments = []
            total_confidence = 0.0
            
            # PaddleOCR结果格式: [det_res, rec_res]
            # det_res: 检测结果，包含文本框坐标
            # rec_res: 识别结果，包含文本和置信度
            
            # 如果没有检测到文本
            if not result or len(result) == 0:
                return OCRResult(full_text="", segments=[], confidence=0.0)
            
            for line in result:
                # 每行结果格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]]
                if len(line) != 2:
                    continue
                
                bbox, (text, confidence) = line
                
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
            raise OCRError(f"解析PaddleOCR结果失败: {str(e)}")

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
            
            # 获取OCR实例
            ocr = self._get_ocr()
            
            # 识别文本
            result = ocr.ocr(
                img=image,
                cls=self.cls,
                det=self.det,
                rec=self.rec
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
        # 由于PaddleOCR不是异步库，我们使用线程池来异步执行
        loop = asyncio.get_event_loop()
        
        async def _arecognize():
            # 在线程池中执行同步方法
            return await loop.run_in_executor(None, self.recognize_text, image_source)
        
        # 使用异步重试机制
        return await self._async_retry_with_backoff(_arecognize)

    def get_supported_languages(self) -> Dict[str, str]:
        """
        获取PaddleOCR支持的语言列表
        
        返回:
            Dict[str, str]: 语言代码到语言名称的映射
        """
        # PaddleOCR支持的语言
        return {
            'ch': '中文',
            'en': '英文',
            'french': '法语',
            'german': '德语',
            'korean': '韩语',
            'japan': '日语',
            'chinese_cht': '繁体中文',
            'ta': '泰米尔语',
            'te': '泰卢固语',
            'ka': '卡纳达语',
            'latin': '拉丁语',
            'arabic': '阿拉伯语',
            'cyrillic': '西里尔语',
            'devanagari': '梵文'
        }

    def get_model_info(self) -> dict:
        """
        返回关于OCR模型的信息
        
        返回:
            dict: 模型信息
        """
        info = super().get_model_info()
        info.update({
            "lang": self.lang,
            "use_gpu": self.use_gpu,
            "use_angle_cls": self.use_angle_cls,
            "det": self.det,
            "rec": self.rec,
            "cls": self.cls,
            "enable_mkldnn": self.enable_mkldnn,
            "cpu_threads": self.cpu_threads,
            "det_model_dir": self.det_model_dir,
            "rec_model_dir": self.rec_model_dir,
            "cls_model_dir": self.cls_model_dir,
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
    
    logger.info("PaddleOCR模型测试")
    
    try:
        # 创建PaddleOCR模型实例
        ocr = PaddleOCRModel(lang='ch', use_gpu=False)
        logger.info(f"模型信息: {ocr.get_model_info()}")
        
        # 注意：实际测试需要有效的图像文件
        # test_image = "path/to/test_image.jpg"
        # result = ocr.recognize_text(test_image)
        # logger.info(f"识别结果: {result.full_text}")
        
    except OCRError as e:
        logger.error(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"未预期的错误: {str(e)}")
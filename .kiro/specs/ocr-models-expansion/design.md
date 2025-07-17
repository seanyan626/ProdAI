# OCR模型扩展设计文档

## 概述

本设计文档描述了如何在ProdAI框架中扩展OCR功能，在现有基础上添加几个主流OCR服务提供商的支持。设计保持简洁，遵循现有框架的模块化原则。

## 架构设计

### 分层的目录结构

```
core/models/ocr/
├── __init__.py
├── base_ocr_model.py          # 基础抽象类（已存在，需小幅增强）
├── openai_ocr_model.py        # OpenAI实现（已存在）
├── cloud/                     # 云服务OCR
│   ├── __init__.py
│   ├── aliyun_ocr_model.py    # 阿里云OCR
│   └── baidu_ocr_model.py     # 百度OCR
├── local/                     # 本地OCR
│   ├── __init__.py
│   ├── paddle_ocr_model.py    # PaddleOCR
│   └── easy_ocr_model.py      # EasyOCR
└── ocr_factory.py             # 简单的工厂类
```

### 核心组件设计

#### 1. 增强现有的基础类

**BaseOCRModel** (小幅增强)
- 添加基本的重试机制
- 统一错误处理

**OCRResult** (小幅增强)
```python
class OCRResult:
    def __init__(self, full_text: str, segments: List[Dict] = None, confidence: float = 0.0):
        self.full_text = full_text
        self.segments = segments or []  # [{"text": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        self.confidence = confidence
```

#### 2. 具体OCR实现

**云服务OCR**
- **AliyunOCRModel**: 直接继承BaseOCRModel，调用阿里云API
- **BaiduOCRModel**: 直接继承BaseOCRModel，调用百度API

**本地OCR**  
- **PaddleOCRModel**: 直接继承BaseOCRModel，使用PaddleOCR库
- **EasyOCRModel**: 直接继承BaseOCRModel，使用EasyOCR库

#### 3. 简单工厂

**OCRFactory**
```python
class OCRFactory:
    @staticmethod
    def create_ocr(provider: str, **kwargs) -> BaseOCRModel:
        if provider == "aliyun":
            return AliyunOCRModel(**kwargs)
        elif provider == "baidu":
            return BaiduOCRModel(**kwargs)
        elif provider == "paddle":
            return PaddleOCRModel(**kwargs)
        elif provider == "easyocr":
            return EasyOCRModel(**kwargs)
        else:
            raise ValueError(f"不支持的OCR提供商: {provider}")
```

### 配置管理

#### 在.env文件中添加OCR配置
```bash
# OCR 云服务配置
ALIYUN_OCR_API_KEY=your_aliyun_ocr_api_key_here
BAIDU_OCR_API_KEY=your_baidu_ocr_api_key_here
BAIDU_OCR_SECRET_KEY=your_baidu_ocr_secret_key_here

# OCR 基础设置
DEFAULT_OCR_PROVIDER=paddle
OCR_TIMEOUT=30
OCR_MAX_RETRIES=3
```

#### 在configs/config.py中添加对应的配置变量
```python
# OCR 基础配置
ALIYUN_OCR_API_KEY: str = None
BAIDU_OCR_API_KEY: str = None
BAIDU_OCR_SECRET_KEY: str = None
DEFAULT_OCR_PROVIDER: str = "paddle"
OCR_TIMEOUT: int = 30
OCR_MAX_RETRIES: int = 3
```

#### 在load_config()函数中添加OCR配置加载
```python
# OCR 配置
g['ALIYUN_OCR_API_KEY'] = os.getenv("ALIYUN_OCR_API_KEY")
g['BAIDU_OCR_API_KEY'] = os.getenv("BAIDU_OCR_API_KEY")
g['BAIDU_OCR_SECRET_KEY'] = os.getenv("BAIDU_OCR_SECRET_KEY")
g['DEFAULT_OCR_PROVIDER'] = os.getenv("DEFAULT_OCR_PROVIDER", "paddle")
g['OCR_TIMEOUT'] = int(os.getenv("OCR_TIMEOUT", "30"))
g['OCR_MAX_RETRIES'] = int(os.getenv("OCR_MAX_RETRIES", "3"))
```

### 错误处理

简单的异常类：
```python
class OCRError(Exception):
    """OCR基础异常"""
    pass
```

## 实现计划

### 第一阶段：基础增强
1. 小幅增强BaseOCRModel和OCRResult类
2. 实现简单的OCRFactory

### 第二阶段：云服务集成
1. 实现AliyunOCRModel
2. 实现BaiduOCRModel

### 第三阶段：本地模型集成
1. 实现PaddleOCRModel
2. 实现EasyOCRModel

这个简化设计专注于核心功能，避免过度工程化，保持代码的简洁性和可维护性。
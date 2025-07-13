# core/models/llm/base_llm.py
# LLM (大语言模型) 交互的抽象基类
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):  # 类名已更改
    """
    LLM (大语言模型) 交互的抽象基类。
    子类应实现与特定LLM服务交互的具体逻辑。
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs: Any):
        """
        初始化 LLM。

        参数:
            model_name (str): 要使用的模型名称。
            api_key (Optional[str]): LLM 服务的 API 密钥。
            **kwargs: 用于 LLM 配置的其他关键字参数。
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        # self._initialize_client() # 初始化客户端的调用通常在子类的构造函数中进行

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        初始化特定的 LLM 客户端 (例如 OpenAI 客户端)。
        此方法应由子类实现。
        """
        pass

    @abstractmethod
    def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:
        """
        为给定的提示生成文本补全。
        （实现待补充）
        """
        pass

    @abstractmethod
    async def agenerate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:
        """
        异步为给定的提示生成文本补全。
        （实现待补充）
        """
        pass

    def chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        为给定的消息序列生成聊天补全。
        （实现待补充）
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持通过此方法直接进行聊天补全。请在子类中实现它。")

    async def achat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        异步为给定的消息序列生成聊天补全。
        （实现待补充）
        """
        raise NotImplementedError(f"{self.__class__.__name__} 类不支持通过此方法直接进行异步聊天补全。请在子类中实现它。")

    def get_model_info(self) -> Dict[str, Any]:
        """
        返回有关模型的信息。
        """
        return {
            "model_name": self.model_name,
            "config": self.config
        }

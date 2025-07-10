# src/llms/base_llm.py
# 大语言模型交互的抽象基类
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLLM(ABC):
    """
    大语言模型 (LLM) 交互的抽象基类。
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
        self._initialize_client()

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

        参数:
            prompt (str): 输入的提示。
            max_tokens (Optional[int]): 要生成的最大 token 数。
            temperature (Optional[float]): 采样温度。
            stop_sequences (Optional[List[str]]): 在何处停止生成的序列。
            **kwargs: 其他特定于提供程序的参数。

        返回:
            str: 生成的文本。
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

        参数:
            prompt (str): 输入的提示。
            max_tokens (Optional[int]): 要生成的最大 token 数。
            temperature (Optional[float]): 采样温度。
            stop_sequences (Optional[List[str]]): 在何处停止生成的序列。
            **kwargs: 其他特定于提供程序的参数。

        返回:
            str: 生成的文本。
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
        这是聊天模型 (例如 GPT-3.5-turbo, GPT-4) 的常见模式。

        参数:
            messages (List[Dict[str, str]]): 消息字典列表，
                例如 [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀！"}]。
            max_tokens (Optional[int]): 要生成的最大 token 数。
            temperature (Optional[float]): 采样温度。
            stop_sequences (Optional[List[str]]): 在何处停止生成的序列。
            **kwargs: 其他特定于提供程序的参数。

        返回:
            Dict[str, Any]: 来自 LLM 的完整响应，通常包括消息内容和其他元数据。
                            示例: {"role": "assistant", "content": "响应文本"}

        注意: 如果子类的底层模型支持聊天补全，则应实现此方法。
              如果不支持，它们可以引发 NotImplementedError 或进行调整。
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

        参数:
            messages (List[Dict[str, str]]): 消息字典列表。
            max_tokens (Optional[int]): 要生成的最大 token 数。
            temperature (Optional[float]): 采样温度。
            stop_sequences (Optional[List[str]]): 在何处停止生成的序列。
            **kwargs: 其他特定于提供程序的参数。

        返回:
            Dict[str, Any]: 来自 LLM 的完整响应。
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

    # 你可以在此处添加其他常用方法，例如:
    # - count_tokens(text: str) -> int: 计算文本中的 token 数量
    # - get_embeddings(texts: List[str]) -> List[List[float]]: 获取文本的嵌入向量 (尽管这可能更适合放在单独的 Embeddings 类中)

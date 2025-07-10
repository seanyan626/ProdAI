# core/memory/base_memory.py
# Agent 使用的记忆系统的抽象基类
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Optional

class BaseMemory(ABC):
    """
    Agent 使用的记忆系统的抽象基类。
    子类应实现具体的记忆存储和检索逻辑。
    """

    @abstractmethod
    def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
        """
        向记忆中添加一条消息或交互。
        （实现待补充）
        """
        pass

    @abstractmethod
    def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索对话历史。
        （实现待补充）
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清除整个记忆。
        （实现待补充）
        """
        pass

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        从记忆中检索最近的一条消息。
        （实现待补充，或依赖 get_history）
        """
        # history = self.get_history(max_messages=1)
        # return history[-1] if history else None
        pass

if __name__ == '__main__':
    # 此处可以添加直接测试此模块内（如果它是具体类的话）或其子类功能的代码
    print("core.memory.base_memory 模块。这是一个抽象基类，通常不直接运行。")
    pass

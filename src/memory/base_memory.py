# src/memory/base_memory.py
# Agent 使用的记忆系统的抽象基类
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Optional # 确保 Optional 被导入

class BaseMemory(ABC):
    """
    Agent 使用的记忆系统的抽象基类。
    """

    @abstractmethod
    def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
        """
        向记忆中添加一条消息或交互。

        参数:
            message (Union[str, Dict[str, Any]]): 消息内容。
                可以是一个简单的字符串，也可以是一个结构化的字典 (例如 OpenAI 消息格式)。
            role (Optional[str]): 与消息关联的角色 (例如 "user", "assistant", "system")。
                                   如果 message 是简单字符串，则此参数为必需。
        """
        pass

    @abstractmethod
    def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索对话历史。

        参数:
            max_messages (Optional[int]): 要检索的最近消息的最大数量。
            max_tokens (Optional[int]): 历史记录应大致对应的最大 token 数。
                                        (实现可能涉及 token 计数和截断)。

        返回:
            List[Dict[str, Any]]: 消息字典列表，通常采用与 LLM 聊天 API 兼容的格式
                                  (例如 [{"role": "user", "content": "..."}, ...])。
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清除整个记忆。
        """
        pass

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        从记忆中检索最近的一条消息。

        返回:
            Optional[Dict[str, Any]]: 最后一条消息，如果记忆为空则为 None。
        """
        history = self.get_history(max_messages=1)
        return history[-1] if history else None

    # 可选: 用于更复杂记忆操作的方法，如摘要或修剪
    # def summarize(self) -> str:
    #     """ 返回对话的摘要。"""
    #     raise NotImplementedError
    #
    # def prune(self, strategy: str = "fifo", max_size: Optional[int] = None) -> None:
    #     """ 根据策略修剪记忆。"""
    #     raise NotImplementedError

if __name__ == '__main__':
    # 具体实现如何进行测试的示例 (概念性)
    class SimpleTestMemory(BaseMemory):
        def __init__(self):
            self.history: List[Dict[str, Any]] = []

        def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
            if isinstance(message, str):
                if not role:
                    raise ValueError("如果消息是字符串，则必须提供角色。")
                self.history.append({"role": role, "content": message})
            elif isinstance(message, dict) and "content" in message and "role" in message:
                self.history.append(message)
            else:
                raise TypeError("消息必须是字符串 (带角色) 或包含 'role' 和 'content' 的字典。")
            print(f"记忆: 已添加消息 - 角色: {self.history[-1]['role']}, 内容: '{self.history[-1]['content'][:50]}...'")

        def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
            # 基本的 max_messages，此简单测试未实现 token 计数
            if max_messages is not None and max_messages > 0:
                return self.history[-max_messages:]
            return list(self.history) # 返回副本

        def clear(self) -> None:
            self.history = []
            print("记忆: 已清除。")

    # 测试 SimpleTestMemory
    print("使用 SimpleTestMemory 实现测试 BaseMemory...")
    memory = SimpleTestMemory()
    memory.add_message("你好！", role="user")
    memory.add_message({"role": "assistant", "content": "嗨！今天我能帮你做些什么？"})
    memory.add_message("给我讲个笑话。", role="user")

    print("\n完整历史:")
    for msg in memory.get_history():
        print(msg)

    print("\n最近 2 条消息:")
    for msg in memory.get_history(max_messages=2):
        print(msg)

    print(f"\n最后一条消息: {memory.get_last_message()}")

    memory.clear()
    print(f"\n清除后的历史: {memory.get_history()}")
    assert not memory.get_history()
    print("BaseMemory 概念性测试完成。")

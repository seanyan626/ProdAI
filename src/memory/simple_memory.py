# src/memory/simple_memory.py
# 简单的内存记忆实现
import logging
from typing import Any, List, Dict, Optional, Union

from .base_memory import BaseMemory

logger = logging.getLogger(__name__)

class SimpleMemory(BaseMemory):
    """
    一个简单的内存存储，用于对话历史。
    它将消息存储为字典列表。
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        初始化简单内存。

        参数:
            system_message (Optional[str]): 一个可选的系统消息，会添加到历史记录的开头。
        """
        self.history: List[Dict[str, Any]] = []
        self.system_message_content: Optional[str] = system_message
        if system_message:
            # 如果提供了系统消息，则添加，但不作为常规“添加”进行记录
            self._prepend_system_message()
        logger.info(f"SimpleMemory 已初始化。系统消息: {'已设置' if system_message else '未设置'}")

    def _prepend_system_message(self):
        """内部方法，用于添加或确保系统消息位于开头。"""
        if not self.system_message_content:
            return

        # 如果存在现有系统消息，则将其删除，以避免在重新初始化或修改时出现重复
        self.history = [msg for msg in self.history if msg.get("role") != "system"]
        self.history.insert(0, {"role": "system", "content": self.system_message_content})


    def add_message(self, message: Union[str, Dict[str, Any]], role: Optional[str] = None) -> None:
        """
        向记忆中添加一条消息。

        参数:
            message (Union[str, Dict[str, Any]]): 消息内容。
                如果是字符串，必须提供 `role`。
                如果是字典，必须包含 "role" 和 "content" 键。
            role (Optional[str]): 消息发送者的角色 (例如 "user", "assistant")。
                                   如果 message 是字典，则忽略此参数。
        """
        if isinstance(message, str):
            if not role:
                logger.error("当消息是字符串时，必须提供角色。")
                raise ValueError("当消息是字符串时，必须提供角色。")
            msg_dict = {"role": role, "content": message}
        elif isinstance(message, dict):
            if "role" not in message or "content" not in message:
                logger.error("消息字典必须包含 'role' 和 'content' 键。")
                raise ValueError("消息字典必须包含 'role' 和 'content' 键。")
            msg_dict = message
        else:
            logger.error(f"不支持的消息类型: {type(message)}")
            raise TypeError("消息必须是字符串或字典。")

        self.history.append(msg_dict)
        logger.debug(f"已添加到 SimpleMemory: 角色: {msg_dict['role']}, 内容: '{str(msg_dict['content'])[:70]}...'")

    def get_history(self, max_messages: Optional[int] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索对话历史。

        参数:
            max_messages (Optional[int]): 要检索的最近消息的最大数量。
                                        如果存在系统消息，则始终包含它，并且不计入 max_messages。
            max_tokens (Optional[int]): (在 SimpleMemory 中未实现) 历史记录的最大 token 数。

        返回:
            List[Dict[str, Any]]: 消息字典列表。
        """
        if max_tokens is not None:
            logger.warning("SimpleMemory 中未实现 max_tokens。将根据 max_messages 返回。")

        # 确保系统消息（如果存在）被保留
        current_history = list(self.history) # 使用副本进行操作
        output_history: List[Dict[str, Any]] = []

        system_msg = None
        if current_history and current_history[0].get("role") == "system":
            system_msg = current_history.pop(0) # 暂时移除系统消息

        if max_messages is not None and max_messages > 0:
            # 从历史记录的非系统部分获取最后 'max_messages' 条消息
            output_history = current_history[-max_messages:]
        else:
            output_history = current_history # 获取所有非系统消息

        if system_msg:
            # 将系统消息添加回开头
            output_history.insert(0, system_msg)

        logger.debug(f"从 SimpleMemory 检索到 {len(output_history)} 条消息。")
        return output_history

    def clear(self) -> None:
        """
        清除内存中的所有消息，但如果设置了初始系统消息，则保留它。
        """
        self.history = []
        if self.system_message_content:
            self._prepend_system_message() # 清除后重新添加系统消息
        logger.info("SimpleMemory 已清除 (如果已配置，则保留系统消息)。")

    def set_system_message(self, system_message: str) -> None:
        """
        设置或更新系统消息。
        """
        self.system_message_content = system_message
        self._prepend_system_message()
        logger.info(f"SimpleMemory 中的系统消息已更新: '{system_message[:70]}...'")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config() # 确保为任何底层组件加载环境变量
    setup_logging() # 为测试配置日志记录

    logger.info("正在测试 SimpleMemory...")

    # 测试无系统消息的情况
    memory1 = SimpleMemory()
    memory1.add_message("你好", role="user")
    memory1.add_message({"role": "assistant", "content": "你好呀！"})
    history1 = memory1.get_history()
    logger.info(f"Memory1 历史 (无系统消息): {history1}")
    assert len(history1) == 2
    assert history1[0]["content"] == "你好"

    # 测试有系统消息的情况
    sys_msg = "你是一个乐于助人的 AI。"
    memory2 = SimpleMemory(system_message=sys_msg)
    memory2.add_message("今天天气怎么样？", role="user")
    memory2.add_message({"role": "assistant", "content": "今天天气晴朗！"})

    history2_full = memory2.get_history()
    logger.info(f"Memory2 完整历史 (有系统消息): {history2_full}")
    assert len(history2_full) == 3
    assert history2_full[0]["role"] == "system"
    assert history2_full[0]["content"] == sys_msg

    history2_limited = memory2.get_history(max_messages=1)
    logger.info(f"Memory2 有限历史 (max_messages=1): {history2_limited}")
    assert len(history2_limited) == 2 # 系统消息 + 1 条用户/助手消息
    assert history2_limited[0]["role"] == "system"
    assert history2_limited[1]["content"] == "今天天气晴朗！"

    last_msg = memory2.get_last_message()
    logger.info(f"Memory2 最后一条消息: {last_msg}")
    assert last_msg and last_msg["content"] == "今天天气晴朗！"

    memory2.clear()
    history_after_clear = memory2.get_history()
    logger.info(f"Memory2 清除后的历史: {history_after_clear}")
    assert len(history_after_clear) == 1 # 应仅包含系统消息
    assert history_after_clear[0]["role"] == "system"

    # 测试更改系统消息
    memory2.set_system_message("你是一个有创造力的 AI。")
    history_new_sys_msg = memory2.get_history()
    logger.info(f"Memory2 更改系统消息后的历史: {history_new_sys_msg}")
    assert len(history_new_sys_msg) == 1
    assert history_new_sys_msg[0]["content"] == "你是一个有创造力的 AI。"
    memory2.add_message("发挥你的创造力！", role="user")
    assert len(memory2.get_history()) == 2


    logger.info("SimpleMemory 测试成功完成。")

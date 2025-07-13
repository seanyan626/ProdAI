# core/tools/base_tool.py
# Agent 可用工具的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolInputSchema(BaseModel):
    """
    对于不需要特定输入的工具的默认空模式。
    BaseTool 的子类可以定义自己的 Pydantic 模型作为输入模式。
    """
    pass


class BaseTool(ABC):
    """
    Agent 可用工具的抽象基类。
    子类应定义工具的具体行为和输入模式。
    """
    name: str = "base_tool"  # 工具名称
    description: str = "一个什么都不做的基础工具。"  # 工具描述
    args_schema: Optional[Type[BaseModel]] = ToolInputSchema  # 输入模式的 Pydantic 模型

    def __init__(self, **kwargs: Any):
        """
        初始化工具。
        （基本实现，子类可扩展）
        """
        self.config = kwargs
        logger.debug(f"工具 '{self.name}' 已使用配置初始化: {kwargs}")

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        工具的核心同步执行逻辑。
        （实现待补充）
        """
        pass

    async def _arun(self, **kwargs: Any) -> Any:
        """
        工具的核心异步执行逻辑。
        （实现待补充，或默认调用同步版本）
        """
        # logger.warning(f"工具 '{self.name}' 未实现异步版本。如果通过 arun 调用，将回退到同步 _run。")
        # return self._run(**kwargs)
        pass

    def run(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        执行工具的公共同步方法。
        （实现待补充，通常包含输入验证和调用 _run）
        """
        # validated_args_dict = {}
        # if self.args_schema:
        #     input_data = tool_input if tool_input is not None else kwargs
        #     try:
        #         # ... (验证逻辑) ...
        #         validated_args_dict = validated_args.model_dump()
        #         logger.info(f"正在使用验证后的输入运行工具 '{self.name}': {validated_args_dict}")
        #     except Exception as e:
        #         logger.error(f"工具 '{self.name}' 的输入验证错误: {e}。输入: {input_data}", exc_info=True)
        #         return f"错误: 工具 {self.name} 的输入验证失败。详情: {e}"
        # else:
        #     validated_args_dict = kwargs if tool_input is None else (tool_input if isinstance(tool_input, dict) else {})
        #     logger.info(f"正在运行工具 '{self.name}' (无模式)，输入: {validated_args_dict}")
        # try:
        #     return self._run(**validated_args_dict)
        # except Exception as e:
        #     logger.error(f"执行工具 '{self.name}' 期间出错: {e}", exc_info=True)
        #     return f"错误: 工具 {self.name} 执行失败。详情: {e}"
        logger.info(f"调用工具 '{self.name}' 的 run 方法。输入（如果有）: {tool_input or kwargs}")
        return f"工具 '{self.name}' 的模拟同步执行结果。"

    async def arun(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        异步执行工具的公共方法。
        （实现待补充，通常包含输入验证和调用 _arun）
        """
        logger.info(f"调用工具 '{self.name}' 的 arun 方法。输入（如果有）: {tool_input or kwargs}")
        return f"工具 '{self.name}' 的模拟异步执行结果。"

    def get_schema_json(self) -> Optional[Dict[str, Any]]:
        """
        返回工具输入参数的 JSON 模式。
        （实现待补充）
        """
        # if self.args_schema:
        #     return self.args_schema.model_json_schema()
        return None  # 占位符

    @classmethod
    def get_tool_info(cls) -> Dict[str, Any]:
        """
        返回包含工具名称、描述和输入模式的字典。
        （实现待补充）
        """
        # schema = None
        # if cls.args_schema:
        #     try:
        #         schema = cls.args_schema.model_json_schema()
        #     except AttributeError:
        #         try:
        #             schema = cls.args_schema.schema()
        #         except Exception as e:
        #             logger.error(f"无法为 {cls.name} 生成模式: {e}")
        return {
            "name": cls.name,
            "description": cls.description,
            "args_schema": None  # 占位符
        }


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
    logger.info("BaseTool 模块可以直接运行测试（如果包含测试代码）。")
    pass

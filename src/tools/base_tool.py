# src/tools/base_tool.py
# Agent 可用工具的抽象基类
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union # 确保 Union 被导入
from pydantic import BaseModel, Field # 确保 Field 被导入

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
    """
    name: str = "base_tool" # 工具名称
    description: str = "一个什么都不做的基础工具。" # 工具描述
    # 输入模式的 Pydantic 模型。子类应重写此项。
    args_schema: Optional[Type[BaseModel]] = ToolInputSchema

    def __init__(self, **kwargs: Any):
        """
        初始化工具。可以接受任意关键字参数
        用于特定于工具的配置。
        """
        # 如果需要，你可以存储 kwargs，或处理它们。
        self.config = kwargs
        logger.debug(f"工具 '{self.name}' 已使用配置初始化: {kwargs}")

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        工具的核心逻辑。此方法必须由子类实现。
        它根据 `args_schema` 接收经过验证的参数。

        参数:
            **kwargs: 任意关键字参数，通常与工具 `args_schema` 中定义的字段匹配。

        返回:
            Any: 工具执行的结果。可以是字符串、
                 字典或任何其他可序列化的类型。
        """
        pass

    async def _arun(self, **kwargs: Any) -> Any:
        """
        核心逻辑的异步版本。
        默认情况下，它会引发 NotImplementedError。如果子类支持异步执行，
        则应重写此方法。
        """
        logger.warning(f"工具 '{self.name}' 未实现异步版本。如果通过 arun 调用，将回退到同步 _run。")
        # 如果未重写，则回退到同步执行，
        # 但这不是真正的异步。
        # 对于真正的异步，此方法需要是 `async def` 并使用 `await`。
        # 考虑使用 `asyncio.to_thread` 在异步上下文中运行同步代码。
        return self._run(**kwargs)


    def run(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        执行工具的公共方法。
        如果提供了 `args_schema`，它会根据该模式验证输入。

        参数:
            tool_input (Optional[Union[Dict[str, Any], BaseModel]]):
                工具的输入。可以是字典或 Pydantic 模型实例。
                如果提供了此参数，则在模式验证时忽略 `kwargs`。
            **kwargs: 如果 `tool_input` 为 None，则这些 kwargs 用作输入。

        返回:
            Any: 来自 `_run` 的结果。
        """
        validated_args_dict = {}
        if self.args_schema:
            input_data = tool_input if tool_input is not None else kwargs
            try:
                if isinstance(input_data, BaseModel):
                    # 如果它已经是正确类型的 pydantic 模型实例
                    if isinstance(input_data, self.args_schema):
                        validated_args = input_data
                    else: # 如果是其他 pydantic 模型，尝试通过字典转换
                        validated_args = self.args_schema(**input_data.model_dump())
                elif isinstance(input_data, dict):
                    validated_args = self.args_schema(**input_data)
                else: # 处理 input_data 为 None 或不是 dict/BaseModel 的情况
                    if input_data is None and not any(self.args_schema.model_fields.values()): # 空模式
                         validated_args = self.args_schema()
                    elif input_data is None and any(f.is_required() for f in self.args_schema.model_fields.values()):
                        raise ValueError(f"工具 '{self.name}' 需要输入，但提供了 None。")
                    elif input_data is None: # 没有必填字段，可以继续使用空输入
                        validated_args = self.args_schema()
                    else:
                        raise TypeError(f"工具输入必须是字典、Pydantic 模型或 None (如果模式允许)。得到 {type(input_data)}")

                validated_args_dict = validated_args.model_dump()
                logger.info(f"正在使用验证后的输入运行工具 '{self.name}': {validated_args_dict}")
            except Exception as e: # 捕获 Pydantic 的 ValidationError 和其他错误
                logger.error(f"工具 '{self.name}' 的输入验证错误: {e}。输入: {input_data}", exc_info=True)
                # 考虑返回错误消息或引发特定异常
                return f"错误: 工具 {self.name} 的输入验证失败。详情: {e}"
        else: # 无模式，直接传递 kwargs (如果没有 kwargs，则传递空字典)
            validated_args_dict = kwargs if tool_input is None else (tool_input if isinstance(tool_input, dict) else {})
            logger.info(f"正在运行工具 '{self.name}' (无模式)，输入: {validated_args_dict}")

        try:
            return self._run(**validated_args_dict)
        except Exception as e:
            logger.error(f"执行工具 '{self.name}' 期间出错: {e}", exc_info=True)
            # 返回结构化错误或引发自定义异常
            return f"错误: 工具 {self.name} 执行失败。详情: {e}"

    async def arun(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        异步执行工具的公共方法。
        验证输入的方式与 `run` 类似。
        """
        validated_args_dict = {}
        if self.args_schema:
            input_data = tool_input if tool_input is not None else kwargs
            try:
                if isinstance(input_data, BaseModel):
                    if isinstance(input_data, self.args_schema):
                        validated_args = input_data
                    else:
                        validated_args = self.args_schema(**input_data.model_dump())
                elif isinstance(input_data, dict):
                    validated_args = self.args_schema(**input_data)
                else:
                    if input_data is None and not any(self.args_schema.model_fields.values()):
                         validated_args = self.args_schema()
                    elif input_data is None and any(f.is_required() for f in self.args_schema.model_fields.values()):
                        raise ValueError(f"工具 '{self.name}' 需要输入，但在异步运行时提供了 None。")
                    elif input_data is None:
                        validated_args = self.args_schema()
                    else:
                        raise TypeError(f"异步工具输入必须是字典、Pydantic 模型或 None。得到 {type(input_data)}")

                validated_args_dict = validated_args.model_dump()
                logger.info(f"正在异步运行工具 '{self.name}'，验证后的输入: {validated_args_dict}")
            except Exception as e:
                logger.error(f"工具 '{self.name}' 的异步输入验证错误: {e}。输入: {input_data}", exc_info=True)
                return f"错误: 工具 {self.name} 的异步输入验证失败。详情: {e}"
        else:
            validated_args_dict = kwargs if tool_input is None else (tool_input if isinstance(tool_input, dict) else {})
            logger.info(f"正在异步运行工具 '{self.name}' (无模式)，输入: {validated_args_dict}")

        try:
            return await self._arun(**validated_args_dict)
        except Exception as e:
            logger.error(f"异步执行工具 '{self.name}' 期间出错: {e}", exc_info=True)
            return f"错误: 异步工具 {self.name} 执行失败。详情: {e}"

    def get_schema_json(self) -> Optional[Dict[str, Any]]:
        """
        返回工具输入参数的 JSON 模式。
        有助于需要理解工具输入的 Agent 或系统。
        """
        if self.args_schema:
            return self.args_schema.model_json_schema()
        return None

    @classmethod
    def get_tool_info(cls) -> Dict[str, Any]:
        """
        返回包含工具名称、描述和输入模式的字典。
        用于向 Agent 提供工具规范。
        """
        schema = None
        if cls.args_schema:
            try:
                # Pydantic v2 使用 model_json_schema()
                schema = cls.args_schema.model_json_schema()
            except AttributeError: # 必要时回退到类似 Pydantic v1 的行为
                try:
                    schema = cls.args_schema.schema()
                except Exception as e:
                    logger.error(f"无法为 {cls.name} 生成模式: {e}")


        return {
            "name": cls.name,
            "description": cls.description,
            "args_schema": schema
        }


if __name__ == '__main__':
    # BaseTool 的示例用法和测试
    from configs.config import load_config
    from configs.logging_config import setup_logging
    import asyncio

    load_config()
    setup_logging()

    logger.info("正在测试 BaseTool...")

    # --- 示例工具 1: 简单回声工具 (无特定模式) ---
    class EchoTool(BaseTool):
        name = "echo_tool"
        description = "回显输入的字符串。"
        # 无特定 args_schema，将使用默认的 ToolInputSchema (空) 或传递 kwargs

        def _run(self, **kwargs: Any) -> str:
            if not kwargs:
                return "回声: 未提供输入。"
            # 为简单起见，将 kwargs 转换为字符串
            return f"回声: {kwargs}"

    echo_tool = EchoTool()
    logger.info(f"EchoTool 信息: {echo_tool.get_tool_info()}")
    result = echo_tool.run(text="你好，世界！")
    logger.info(f"EchoTool 结果: {result}")
    assert "你好，世界！" in result
    result_no_input = echo_tool.run() # 测试无输入的情况
    logger.info(f"EchoTool 结果 (无输入): {result_no_input}")
    assert "未提供输入" in result_no_input


    # --- 示例工具 2: 加法器工具 (带 Pydantic 模式) ---
    class AdderToolInput(BaseModel):
        a: int = Field(..., description="要相加的第一个数字")
        b: int = Field(..., description="要相加的第二个数字")
        c: Optional[int] = Field(0, description="可选的第三个要相加的数字")

    class AdderTool(BaseTool):
        name = "adder_tool"
        description = "将两个或三个数字相加。"
        args_schema: Type[BaseModel] = AdderToolInput

        def _run(self, a: int, b: int, c: Optional[int] = 0) -> int:
            return a + b + (c if c is not None else 0)

        async def _arun(self, a: int, b: int, c: Optional[int] = 0) -> int:
            logger.info(f"AdderTool 异步: 正在计算 {a} + {b} + {c}")
            await asyncio.sleep(0.01) # 模拟一些异步工作
            return a + b + (c if c is not None else 0)


    adder_tool = AdderTool()
    logger.info(f"AdderTool 信息: {adder_tool.get_tool_info()}")
    logger.info(f"AdderTool Schema JSON: {adder_tool.get_schema_json()}")

    # 测试有效输入 (作为字典)
    add_result_dict = adder_tool.run({"a": 5, "b": 10})
    logger.info(f"AdderTool 结果 (字典输入): {add_result_dict}")
    assert add_result_dict == 15

    # 测试有效输入 (作为 kwargs)
    add_result_kwargs = adder_tool.run(a=3, b=7, c=2)
    logger.info(f"AdderTool 结果 (kwargs 输入): {add_result_kwargs}")
    assert add_result_kwargs == 12

    # 测试有效输入 (作为 Pydantic 模型实例)
    input_model = AdderToolInput(a=1, b=2, c=3)
    add_result_model = adder_tool.run(input_model)
    logger.info(f"AdderTool 结果 (模型输入): {add_result_model}")
    assert add_result_model == 6

    # 测试无效输入 (缺少必填字段)
    invalid_result = adder_tool.run({"a": 5}) # 缺少 'b'
    logger.info(f"AdderTool 结果 (无效输入): {invalid_result}")
    assert "输入验证失败" in str(invalid_result) # 检查中文错误信息

    # 测试无效输入 (类型错误)
    invalid_type_result = adder_tool.run({"a": "五", "b": 10}) # "五" 是字符串
    logger.info(f"AdderTool 结果 (无效类型): {invalid_type_result}")
    assert "输入验证失败" in str(invalid_type_result)

    # 测试异步运行
    async def run_async_adder():
        async_add_result = await adder_tool.arun(a=10, b=20, c=5)
        logger.info(f"AdderTool 异步结果: {async_add_result}")
        assert async_add_result == 35

        async_invalid_result = await adder_tool.arun({"a": "十"}) # "十" 是字符串
        logger.info(f"AdderTool 异步结果 (无效): {async_invalid_result}")
        assert "异步输入验证失败" in str(async_invalid_result) # 检查中文错误信息

    asyncio.run(run_async_adder())

    logger.info("BaseTool 测试成功完成。")

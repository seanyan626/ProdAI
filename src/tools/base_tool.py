# src/tools/base_tool.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

class ToolInputSchema(BaseModel):
    """
    Default empty schema for tools that don't require specific input.
    Subclasses of BaseTool can define their own Pydantic model for input schema.
    """
    pass

class BaseTool(ABC):
    """
    Abstract base class for tools that an agent can use.
    """
    name: str = "base_tool"
    description: str = "A base tool that does nothing."
    # Pydantic model for input schema. Subclasses should override this.
    args_schema: Optional[Type[BaseModel]] = ToolInputSchema

    def __init__(self, **kwargs: Any):
        """
        Initializes the tool. Can accept arbitrary keyword arguments
        for tool-specific configuration.
        """
        # You can store kwargs if needed, or process them.
        self.config = kwargs
        logger.debug(f"Tool '{self.name}' initialized with config: {kwargs}")

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        The core logic of the tool. This method must be implemented by subclasses.
        It receives validated arguments according to the `args_schema`.

        Args:
            **kwargs: Arbitrary keyword arguments, typically matching the fields
                      defined in the tool's `args_schema`.

        Returns:
            Any: The result of the tool's execution. This can be a string,
                 a dictionary, or any other serializable type.
        """
        pass

    async def _arun(self, **kwargs: Any) -> Any:
        """
        Asynchronous version of the core logic.
        By default, it raises NotImplementedError. Subclasses should override
        this if they support asynchronous execution.
        """
        logger.warning(f"Tool '{self.name}' does not have an async version implemented. Falling back to synchronous _run if called via arun.")
        # Fallback to synchronous execution if not overridden,
        # but this is not truly async.
        # For true async, this method needs to be `async def` and use `await`.
        # Consider using `asyncio.to_thread` for running sync code in async context.
        return self._run(**kwargs)


    def run(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        Public method to execute the tool.
        It validates the input against `args_schema` if provided.

        Args:
            tool_input (Optional[Union[Dict[str, Any], BaseModel]]):
                The input for the tool. Can be a dictionary or a Pydantic model instance.
                If this is provided, `kwargs` are ignored for schema validation.
            **kwargs: If `tool_input` is None, these kwargs are used as input.

        Returns:
            Any: The result from `_run`.
        """
        validated_args_dict = {}
        if self.args_schema:
            input_data = tool_input if tool_input is not None else kwargs
            try:
                if isinstance(input_data, BaseModel):
                    # If it's already a pydantic model instance of the correct type
                    if isinstance(input_data, self.args_schema):
                        validated_args = input_data
                    else: # If it's some other pydantic model, try to convert by dict
                        validated_args = self.args_schema(**input_data.model_dump())
                elif isinstance(input_data, dict):
                    validated_args = self.args_schema(**input_data)
                else: # Handle cases where input_data is None or not a dict/BaseModel
                    if input_data is None and not any(self.args_schema.model_fields.values()): # Empty schema
                         validated_args = self.args_schema()
                    elif input_data is None and any(f.is_required() for f in self.args_schema.model_fields.values()):
                        raise ValueError(f"Tool '{self.name}' requires input, but None was provided.")
                    elif input_data is None: # No required fields, can proceed with empty
                        validated_args = self.args_schema()
                    else:
                        raise TypeError(f"Tool input must be a dictionary, Pydantic model, or None (if schema allows). Got {type(input_data)}")

                validated_args_dict = validated_args.model_dump()
                logger.info(f"Running tool '{self.name}' with validated input: {validated_args_dict}")
            except Exception as e: # Catches Pydantic's ValidationError and others
                logger.error(f"Input validation error for tool '{self.name}': {e}. Input: {input_data}", exc_info=True)
                # Consider returning an error message or raising a specific exception
                return f"Error: Input validation failed for tool {self.name}. Details: {e}"
        else: # No schema, pass kwargs directly (or empty dict if no kwargs)
            validated_args_dict = kwargs if tool_input is None else (tool_input if isinstance(tool_input, dict) else {})
            logger.info(f"Running tool '{self.name}' (no schema) with input: {validated_args_dict}")

        try:
            return self._run(**validated_args_dict)
        except Exception as e:
            logger.error(f"Error during execution of tool '{self.name}': {e}", exc_info=True)
            # Return a structured error or raise a custom exception
            return f"Error: Tool {self.name} execution failed. Details: {e}"

    async def arun(self, tool_input: Optional[Union[Dict[str, Any], BaseModel]] = None, **kwargs: Any) -> Any:
        """
        Public method to asynchronously execute the tool.
        Validates input similarly to `run`.
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
                        raise ValueError(f"Tool '{self.name}' requires input, but None was provided for async run.")
                    elif input_data is None:
                        validated_args = self.args_schema()
                    else:
                        raise TypeError(f"Async tool input must be a dictionary, Pydantic model, or None. Got {type(input_data)}")

                validated_args_dict = validated_args.model_dump()
                logger.info(f"Async running tool '{self.name}' with validated input: {validated_args_dict}")
            except Exception as e:
                logger.error(f"Async input validation error for tool '{self.name}': {e}. Input: {input_data}", exc_info=True)
                return f"Error: Async Input validation failed for tool {self.name}. Details: {e}"
        else:
            validated_args_dict = kwargs if tool_input is None else (tool_input if isinstance(tool_input, dict) else {})
            logger.info(f"Async running tool '{self.name}' (no schema) with input: {validated_args_dict}")

        try:
            return await self._arun(**validated_args_dict)
        except Exception as e:
            logger.error(f"Error during async execution of tool '{self.name}': {e}", exc_info=True)
            return f"Error: Async Tool {self.name} execution failed. Details: {e}"

    def get_schema_json(self) -> Optional[Dict[str, Any]]:
        """
        Returns the JSON schema of the tool's input arguments.
        Helpful for agents or systems that need to understand tool inputs.
        """
        if self.args_schema:
            return self.args_schema.model_json_schema()
        return None

    @classmethod
    def get_tool_info(cls) -> Dict[str, Any]:
        """
        Returns a dictionary with the tool's name, description, and input schema.
        Useful for providing tool specifications to an agent.
        """
        schema = None
        if cls.args_schema:
            try:
                # Pydantic v2 uses model_json_schema()
                schema = cls.args_schema.model_json_schema()
            except AttributeError: # Fallback for Pydantic v1-like behavior if necessary
                try:
                    schema = cls.args_schema.schema()
                except Exception as e:
                    logger.error(f"Could not generate schema for {cls.name}: {e}")


        return {
            "name": cls.name,
            "description": cls.description,
            "args_schema": schema
        }


if __name__ == '__main__':
    # Example usage and test for BaseTool
    from configs.config import load_config
    from configs.logging_config import setup_logging
    import asyncio

    load_config()
    setup_logging()

    logger.info("Testing BaseTool...")

    # --- Example Tool 1: Simple Echo Tool (no specific schema) ---
    class EchoTool(BaseTool):
        name = "echo_tool"
        description = "Echoes back the input string."
        # No specific args_schema, will use the default ToolInputSchema (empty) or pass through kwargs

        def _run(self, **kwargs: Any) -> str:
            if not kwargs:
                return "Echo: No input provided."
            # Convert kwargs to string for simplicity
            return f"Echo: {kwargs}"

    echo_tool = EchoTool()
    logger.info(f"EchoTool Info: {echo_tool.get_tool_info()}")
    result = echo_tool.run(text="Hello, world!")
    logger.info(f"EchoTool result: {result}")
    assert "Hello, world!" in result
    result_no_input = echo_tool.run() # Test with no input
    logger.info(f"EchoTool result (no input): {result_no_input}")
    assert "No input provided" in result_no_input


    # --- Example Tool 2: Adder Tool (with Pydantic schema) ---
    class AdderToolInput(BaseModel):
        a: int = Field(..., description="First number to add")
        b: int = Field(..., description="Second number to add")
        c: Optional[int] = Field(0, description="Optional third number to add")

    class AdderTool(BaseTool):
        name = "adder_tool"
        description = "Adds two or three numbers together."
        args_schema: Type[BaseModel] = AdderToolInput

        def _run(self, a: int, b: int, c: Optional[int] = 0) -> int:
            return a + b + (c if c is not None else 0)

        async def _arun(self, a: int, b: int, c: Optional[int] = 0) -> int:
            logger.info(f"AdderTool async: Adding {a} + {b} + {c}")
            await asyncio.sleep(0.01) # Simulate some async work
            return a + b + (c if c is not None else 0)


    adder_tool = AdderTool()
    logger.info(f"AdderTool Info: {adder_tool.get_tool_info()}")
    logger.info(f"AdderTool Schema JSON: {adder_tool.get_schema_json()}")

    # Test valid input (as dict)
    add_result_dict = adder_tool.run({"a": 5, "b": 10})
    logger.info(f"AdderTool result (dict input): {add_result_dict}")
    assert add_result_dict == 15

    # Test valid input (as kwargs)
    add_result_kwargs = adder_tool.run(a=3, b=7, c=2)
    logger.info(f"AdderTool result (kwargs input): {add_result_kwargs}")
    assert add_result_kwargs == 12

    # Test valid input (as Pydantic model instance)
    input_model = AdderToolInput(a=1, b=2, c=3)
    add_result_model = adder_tool.run(input_model)
    logger.info(f"AdderTool result (model input): {add_result_model}")
    assert add_result_model == 6

    # Test invalid input (missing required field)
    invalid_result = adder_tool.run({"a": 5}) # Missing 'b'
    logger.info(f"AdderTool result (invalid input): {invalid_result}")
    assert "Error: Input validation failed" in str(invalid_result)

    # Test invalid input (wrong type)
    invalid_type_result = adder_tool.run({"a": "five", "b": 10})
    logger.info(f"AdderTool result (invalid type): {invalid_type_result}")
    assert "Error: Input validation failed" in str(invalid_type_result)

    # Test async run
    async def run_async_adder():
        async_add_result = await adder_tool.arun(a=10, b=20, c=5)
        logger.info(f"AdderTool async result: {async_add_result}")
        assert async_add_result == 35

        async_invalid_result = await adder_tool.arun({"a": "ten"})
        logger.info(f"AdderTool async result (invalid): {async_invalid_result}")
        assert "Error: Async Input validation failed" in str(async_invalid_result)

    asyncio.run(run_async_adder())

    logger.info("BaseTool tests completed successfully.")

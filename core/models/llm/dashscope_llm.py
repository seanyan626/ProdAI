# core/models/llm/dashscope_llm.py
# 阿里云通义千问 (DashScope) LLM 接口实现
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from http import HTTPStatus

import dashscope
from dashscope.api_entities.dashscope_response import GenerationResponse

from .base_llm import BaseLLM
from configs.config import DASHSCOPE_API_KEY, DASHSCOPE_API_URL, load_config

# 确保配置已加载
load_config()

logger = logging.getLogger(__name__)

def _convert_dict_messages_to_dashscope(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """将我们内部的消息格式转换为 DashScope 需要的格式。"""
    valid_roles = {"user", "assistant", "system"}
    ds_messages = []
    for msg in messages:
        role = msg.get("role")
        if role not in valid_roles:
            logger.warning(f"DashScope 不支持角色 '{role}'，将替换为 'user'。")
            role = "user"
        ds_messages.append({"role": role, "content": msg.get("content", "")})
    return ds_messages

class DashScopeLLM(BaseLLM):
    """
    使用 DashScope SDK 封装通义千问系列模型。
    """
    def __init__(
        self,
        model_name: str = "qwen-turbo", # 默认使用 qwen-turbo
        api_key: Optional[str] = DASHSCOPE_API_KEY,
        base_url: Optional[str] = DASHSCOPE_API_URL,
        **kwargs: Any # 其他传递给 DashScope API 的参数
    ):
        all_kwargs = {"base_url": base_url, **kwargs}
        super().__init__(model_name=model_name, api_key=api_key, **all_kwargs)
        self.client_initialized = False
        self._initialize_client()

    def _initialize_client(self) -> None:
        """
        初始化 DashScope API 客户端。
        主要是设置 API Key 和可选的 API Host。
        """
        if not self.api_key:
            logger.error("DashScope API 密钥 (DASHSCOPE_API_KEY) 未设置。")
            raise ValueError("DashScope API 密钥缺失。")
        try:
            # 如果提供了 base_url，则通过环境变量设置它，因为 dashscope SDK 主要通过环境变量读取
            base_url = self.config.get("base_url")
            if base_url:
                import os
                os.environ['DASHSCOPE_API_HOST'] = base_url
                logger.info(f"已将 DashScope API Host 设置为: {base_url}")

            dashscope.api_key = self.api_key
            self.client_initialized = True
            logger.info("DashScope API 密钥已设置。")
        except Exception as e:
            logger.error(f"设置 DashScope 配置失败: {e}", exc_info=True)
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        通过调用 chat 方法来为单个提示生成文本补全。
        """
        messages = [{"role": "user", "content": prompt}]
        chat_response = self.chat(messages, max_tokens, temperature, stop_sequences, **kwargs)
        return chat_response.get("content", "")

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        通过调用 achat 方法来异步为单个提示生成文本补全。
        """
        messages = [{"role": "user", "content": prompt}]
        chat_response = await self.achat(messages, max_tokens, temperature, stop_sequences, **kwargs)
        return chat_response.get("content", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        使用 DashScope Generation API 进行聊天补全。
        """
        if not self.client_initialized:
            raise RuntimeError("DashScope 客户端未初始化。")

        ds_messages = _convert_dict_messages_to_dashscope(messages)

        api_params = self.config.copy()
        api_params.update(kwargs)
        if temperature is not None:
            api_params["temperature"] = temperature
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        if stop_sequences is not None:
            api_params["stop"] = stop_sequences

        try:
            logger.debug(f"调用 DashScope Generation API。模型: {self.model_name}。消息数: {len(ds_messages)}。")
            response: GenerationResponse = dashscope.Generation.call(
                model=self.model_name,
                messages=ds_messages,
                result_format='message',
                **api_params
            )

            if response.status_code == HTTPStatus.OK:
                choice = response.output.choices[0]
                content = choice.message.content
                role = choice.message.role

                return_dict = {"role": role, "content": content}
                if response.usage:
                    return_dict["token_usage"] = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                    }
                return return_dict
            else:
                logger.error(f"DashScope API 调用出错: Code: {response.code}, Message: {response.message}")
                return {"role": "assistant", "content": f"错误: API 调用失败 - {response.message}", "error": response.message}

        except Exception as e:
            logger.error(f"调用 DashScope API 失败: {e}", exc_info=True)
            return {"role": "assistant", "content": f"错误: 调用 SDK 失败 - {e}", "error": str(e)}

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        使用 DashScope Generation API 异步进行聊天补全。
        注意：DashScope Python SDK (截至v1.14.0) 的 call 方法本身是同步阻塞的。
        这里我们用 asyncio.to_thread 模拟异步调用。
        """
        import asyncio

        try:
            return await asyncio.to_thread(
                self.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                **kwargs
            )
        except Exception as e:
            logger.error(f"异步调用 DashScope chat 方法失败: {e}", exc_info=True)
            return {"role": "assistant", "content": f"错误: 异步调用失败 - {e}", "error": str(e)}

if __name__ == '__main__':
    import asyncio
    from configs.logging_config import setup_logging

    setup_logging()

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY":
        logger.warning("DashScope API 密钥 (DASHSCOPE_API_KEY) 未配置，跳过 __main__ 测试。")
    else:
        logger.info("测试 DashScopeLLM...")
        try:
            llm = DashScopeLLM(model_name="qwen-turbo")
            logger.info(f"LLM 实例创建成功: {llm.get_model_info()}")

            logger.info("\n--- 测试同步 Chat ---")
            chat_messages = [
                {"role": "user", "content": "你好，用中文介绍一下你自己。"}
            ]
            sync_response = llm.chat(chat_messages)
            logger.info(f"同步 Chat 响应: {sync_response}")
            assert sync_response.get("content") and "通义千问" in sync_response.get("content", "")

            async def run_async_test():
                logger.info("\n--- 测试异步 Chat ---")
                async_response = await llm.achat(chat_messages)
                logger.info(f"异步 Chat 响应: {async_response}")
                assert async_response.get("content") and "通义千问" in async_response.get("content", "")

            asyncio.run(run_async_test())
            logger.info("异步测试完成。")

        except ValueError as ve:
            logger.error(f"初始化或配置错误: {ve}")
        except Exception as e:
            logger.error(f"执行 DashScopeLLM 测试时发生意外错误: {e}", exc_info=True)

        logger.info("DashScopeLLM __main__ 测试结束。")
    pass

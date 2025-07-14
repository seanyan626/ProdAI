# core/models/llm/deepseek_llm.py
# DeepSeek LLM 接口实现
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

from configs.config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, load_config
from .base_llm import BaseLLM

# 确保配置已加载
load_config()

logger = logging.getLogger(__name__)


def _convert_dict_messages_to_langchain(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """将字典格式的消息列表转换为 Langchain BaseMessage 对象列表。"""
    lc_messages = []
    for msg_dict in messages:
        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            logger.warning(f"未知的消息角色: {role}，将作为 HumanMessage 处理。")
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


class DeepSeekLLM(BaseLLM):
    """
    使用 Langchain 封装 DeepSeek LLM。
    由于 DeepSeek API 与 OpenAI API 兼容，我们复用 ChatOpenAI 客户端。
    """

    def __init__(
            self,
            model_name: str = "deepseek-chat",
            api_key: Optional[str] = DEEPSEEK_API_KEY,
            base_url: Optional[str] = DEEPSEEK_API_URL,
            temperature: Optional[float] = 0.7,
            max_tokens: Optional[int] = 2048,
            **kwargs: Any
    ):
        all_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
            **kwargs
        }
        super().__init__(model_name=model_name, api_key=api_key, **all_kwargs)

        self.client: Optional[ChatOpenAI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.api_key:
            logger.error("DeepSeek API 密钥 (DEEPSEEK_API_KEY) 未设置。")
            raise ValueError("DeepSeek API 密钥缺失。")
        try:
            client_params = {
                "model_name": self.model_name,
                "openai_api_key": self.api_key,
                "openai_api_base": self.config.get("base_url"),  # Langchain 使用 openai_api_base
                "temperature": self.config.get("temperature"),
                "max_tokens": self.config.get("max_tokens"),
                **(self.config.get("llm_specific_kwargs") or {})
            }
            client_params = {k: v for k, v in client_params.items() if v is not None}
            self.client = ChatOpenAI(**client_params)
            logger.info(
                f"Langchain ChatOpenAI (用于 DeepSeek) 客户端已为模型 {self.model_name} 初始化。Base URL: {self.config.get('base_url') or '默认'}")
        except Exception as e:
            logger.error(f"初始化 Langchain ChatOpenAI (用于 DeepSeek) 客户端失败: {e}", exc_info=True)
            raise

    def _get_configured_client(
            self,
            max_tokens_runtime: Optional[int] = None,
            temperature_runtime: Optional[float] = None,
            stop_sequences_runtime: Optional[List[str]] = None,
            **kwargs_runtime: Any
    ) -> ChatOpenAI:
        if not self.client:
            raise RuntimeError("LLM 客户端未初始化。")

        binding_kwargs = {}
        if stop_sequences_runtime:
            binding_kwargs["stop"] = stop_sequences_runtime
        if max_tokens_runtime is not None:
            binding_kwargs["max_tokens"] = max_tokens_runtime

        if kwargs_runtime:
            binding_kwargs.update(kwargs_runtime)

        if temperature_runtime is not None and temperature_runtime != self.config.get("temperature"):
            return self.client.bind(temperature=temperature_runtime, **binding_kwargs)
        elif binding_kwargs:
            return self.client.bind(**binding_kwargs)

        return self.client

    def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:

        configured_client = self._get_configured_client(max_tokens, temperature, stop_sequences, **kwargs)
        lc_messages = [HumanMessage(content=prompt)]

        try:
            response_message = configured_client.invoke(lc_messages)
            return response_message.content.strip() if isinstance(response_message.content, str) else str(
                response_message.content)
        except Exception as e:
            logger.error(f"DeepSeek LLM generate 调用失败: {e}", exc_info=True)
            return f"错误：LLM 生成失败 - {e}"

    async def agenerate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:
        configured_client = self._get_configured_client(max_tokens, temperature, stop_sequences, **kwargs)
        lc_messages = [HumanMessage(content=prompt)]

        try:
            response_message = await configured_client.ainvoke(lc_messages)
            return response_message.content.strip() if isinstance(response_message.content, str) else str(
                response_message.content)
        except Exception as e:
            logger.error(f"DeepSeek LLM agenerate 调用失败: {e}", exc_info=True)
            return f"错误：LLM 异步生成失败 - {e}"

    def chat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        configured_client = self._get_configured_client(max_tokens, temperature, stop_sequences, **kwargs)
        lc_messages = _convert_dict_messages_to_langchain(messages)

        try:
            response_message = configured_client.invoke(lc_messages)
            content_str = response_message.content.strip() if isinstance(response_message.content, str) else str(
                response_message.content)

            return_dict = {"role": "assistant", "content": content_str}
            if hasattr(response_message, 'response_metadata') and response_message.response_metadata:
                return_dict["metadata"] = response_message.response_metadata
                if "token_usage" in response_message.response_metadata:
                    return_dict["token_usage"] = response_message.response_metadata["token_usage"]
            return return_dict
        except Exception as e:
            logger.error(f"DeepSeek LLM chat 调用失败: {e}", exc_info=True)
            return {"role": "assistant", "content": f"错误：LLM 聊天失败 - {e}", "error": str(e)}

    async def achat(
            self,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        configured_client = self._get_configured_client(max_tokens, temperature, stop_sequences, **kwargs)
        lc_messages = _convert_dict_messages_to_langchain(messages)

        try:
            response_message = await configured_client.ainvoke(lc_messages)
            content_str = response_message.content.strip() if isinstance(response_message.content, str) else str(
                response_message.content)

            return_dict = {"role": "assistant", "content": content_str}
            if hasattr(response_message, 'response_metadata') and response_message.response_metadata:
                return_dict["metadata"] = response_message.response_metadata
                if "token_usage" in response_message.response_metadata:
                    return_dict["token_usage"] = response_message.response_metadata["token_usage"]
            return return_dict
        except Exception as e:
            logger.error(f"DeepSeek LLM achat 调用失败: {e}", exc_info=True)
            return {"role": "assistant", "content": f"错误：LLM 异步聊天失败 - {e}", "error": str(e)}


if __name__ == '__main__':
    from configs.logging_config import setup_logging

    setup_logging()

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        logger.warning("DeepSeek API 密钥 (DEEPSEEK_API_KEY) 未配置，跳过 __main__ 测试。")
    else:
        logger.info("测试 DeepSeekLLM...")
        try:
            llm = DeepSeekLLM()
            logger.info(f"LLM 实例创建成功: {llm.get_model_info()}")

            logger.info("\n--- 测试同步 Chat ---")
            chat_messages = [{"role": "user", "content": "你好，请用中文介绍一下你自己。"}]
            sync_response = llm.chat(chat_messages)
            logger.info(f"同步 Chat 响应: {sync_response}")
            assert sync_response.get("content")

        except Exception as e:
            logger.error(f"执行 DeepSeekLLM 测试时发生意外错误: {e}", exc_info=True)

        logger.info("DeepSeekLLM __main__ 测试结束。")
    pass

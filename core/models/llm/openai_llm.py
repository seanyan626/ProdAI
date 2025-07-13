# core/models/llm/openai_llm.py
# OpenAI LLM 接口实现 (使用 Langchain)
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

from .base_llm import BaseLLM
from configs.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, load_config

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

class OpenAILLM(BaseLLM):
    """
    使用 Langchain 封装 OpenAI LLM (GPT-3.5-turbo, GPT-4 等)。
    """
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        **kwargs: Any
    ):
        all_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        super().__init__(model_name=model_name, api_key=api_key, **all_kwargs)

        self.client: Optional[ChatOpenAI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        if not self.api_key:
            logger.error("OpenAI API 密钥未设置。请提供密钥或在 .env 文件中设置 OPENAI_API_KEY。")
            raise ValueError("OpenAI API 密钥缺失。")
        try:
            client_params = {
                "model_name": self.model_name,
                "openai_api_key": self.api_key,
                "temperature": self.config.get("temperature", DEFAULT_TEMPERATURE),
                "max_tokens": self.config.get("max_tokens", DEFAULT_MAX_TOKENS),
                **(self.config.get("llm_specific_kwargs") or {})
            }
            client_params = {k: v for k, v in client_params.items() if v is not None}
            self.client = ChatOpenAI(**client_params)
            logger.info(f"Langchain ChatOpenAI 客户端已为模型 {self.model_name} 初始化。")
        except Exception as e:
            logger.error(f"初始化 Langchain ChatOpenAI 客户端失败: {e}", exc_info=True)
            raise

    def _get_configured_client(
        self,
        max_tokens_runtime: Optional[int] = None,
        temperature_runtime: Optional[float] = None,
        stop_sequences_runtime: Optional[List[str]] = None,
        **kwargs_runtime: Any
    ) -> ChatOpenAI:
        """根据运行时参数获取配置好的 Langchain 客户端。"""
        if not self.client:
            logger.error("客户端未初始化。")
            raise RuntimeError("LLM 客户端未初始化。")

        binding_kwargs = {}
        if stop_sequences_runtime:
            binding_kwargs["stop"] = stop_sequences_runtime
        if max_tokens_runtime is not None:
             binding_kwargs["max_tokens"] = max_tokens_runtime

        if kwargs_runtime:
            binding_kwargs.update(kwargs_runtime)

        if temperature_runtime is not None and temperature_runtime != self.config.get("temperature"):
            logger.info(f"为本次调用将 temperature 从 {self.config.get('temperature')} 临时绑定为 {temperature_runtime}。")
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
            logger.debug(
                f"Langchain 调用 generate (实为 chat invoke)。模型: {self.model_name}。提示: '{prompt[:100]}...'。"
            )
            response_message = configured_client.invoke(lc_messages)
            generated_text = response_message.content.strip() if isinstance(response_message.content, str) else str(response_message.content)
            logger.debug(f"Langchain OpenAI 响应: '{generated_text[:100]}...'")
            return generated_text
        except Exception as e:
            logger.error(f"Langchain OpenAI generate 调用失败: {e}", exc_info=True)
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
            logger.debug(
                f"Langchain 异步调用 agenerate (实为 chat ainvoke)。模型: {self.model_name}。提示: '{prompt[:100]}...'。"
            )
            response_message = await configured_client.ainvoke(lc_messages)
            generated_text = response_message.content.strip() if isinstance(response_message.content, str) else str(response_message.content)
            logger.debug(f"Langchain OpenAI 异步响应: '{generated_text[:100]}...'")
            return generated_text
        except Exception as e:
            logger.error(f"Langchain OpenAI agenerate 调用失败: {e}", exc_info=True)
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
            logger.debug(
                f"Langchain 调用 chat (invoke)。模型: {self.model_name}。消息数: {len(lc_messages)}。"
            )
            response_message = configured_client.invoke(lc_messages)
            content_str = response_message.content.strip() if isinstance(response_message.content, str) else str(response_message.content)
            logger.debug(f"Langchain OpenAI 聊天响应: 角色: assistant, 内容: '{content_str[:100]}...'")

            return_dict = {"role": "assistant", "content": content_str}
            if hasattr(response_message, 'response_metadata') and response_message.response_metadata:
                return_dict["metadata"] = response_message.response_metadata
                if "token_usage" in response_message.response_metadata:
                     return_dict["token_usage"] = response_message.response_metadata["token_usage"]
            return return_dict
        except Exception as e:
            logger.error(f"Langchain OpenAI chat 调用失败: {e}", exc_info=True)
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
            logger.debug(
                f"Langchain 异步调用 achat (ainvoke)。模型: {self.model_name}。消息数: {len(lc_messages)}。"
            )
            response_message = await configured_client.ainvoke(lc_messages)
            content_str = response_message.content.strip() if isinstance(response_message.content, str) else str(response_message.content)
            logger.debug(f"Langchain OpenAI 异步聊天响应: 角色: assistant, 内容: '{content_str[:100]}...'")

            return_dict = {"role": "assistant", "content": content_str}
            if hasattr(response_message, 'response_metadata') and response_message.response_metadata:
                return_dict["metadata"] = response_message.response_metadata
                if "token_usage" in response_message.response_metadata:
                     return_dict["token_usage"] = response_message.response_metadata["token_usage"]
            return return_dict
        except Exception as e:
            logger.error(f"Langchain OpenAI achat 调用失败: {e}", exc_info=True)
            return {"role": "assistant", "content": f"错误：LLM 异步聊天失败 - {e}", "error": str(e)}

if __name__ == '__main__':
    import asyncio
    from configs.logging_config import setup_logging
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OpenAI API 密钥未在 .env 文件中配置或为占位符，跳过 Langchain OpenAILLM 的 __main__ 测试。")
    else:
        logger.info("测试 Langchain 封装的 OpenAILLM...")

        try:
            llm = OpenAILLM(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150
            )
            logger.info(f"LLM 实例创建成功: {llm.get_model_info()}")

            logger.info("\n--- 测试同步 Chat ---")
            chat_messages_sync = [
                {"role": "system", "content": "你是一个乐于助人的AI助手。"},
                {"role": "user", "content": "你好，请问法国的首都是哪里？"}
            ]
            sync_chat_response = llm.chat(chat_messages_sync, stop_sequences=["\nHuman:"], temperature=0.1) # 示例运行时覆盖 temperature
            logger.info(f"同步 Chat 响应: {sync_chat_response}")
            assert "巴黎" in sync_chat_response.get("content", "").lower() or "paris" in sync_chat_response.get("content", "").lower()

            logger.info("\n--- 测试同步 Generate (通过 Chat 实现) ---")
            sync_generate_prompt = "简单介绍一下Python语言的特点。"
            sync_generate_response = llm.generate(sync_generate_prompt, max_tokens=100)
            logger.info(f"同步 Generate 响应: {sync_generate_response}")
            assert len(sync_generate_response) > 10

            async def run_async_tests():
                logger.info("\n--- 测试异步 Chat ---")
                chat_messages_async = [
                    {"role": "user", "content": "用一句话描述量子计算。"}
                ]
                async_chat_response = await llm.achat(chat_messages_async, max_tokens=50)
                logger.info(f"异步 Chat 响应: {async_chat_response}")
                assert async_chat_response.get("content")

                logger.info("\n--- 测试异步 Generate (通过 Chat 实现) ---")
                async_generate_prompt = "什么是人工智能？"
                async_generate_response = await llm.agenerate(async_generate_prompt)
                logger.info(f"异步 Generate 响应: {async_generate_response}")
                assert async_generate_response

            logger.info("\n开始异步测试...")
            asyncio.run(run_async_tests())
            logger.info("异步测试完成。")

        except ValueError as ve:
            logger.error(f"初始化或配置错误: {ve}")
        except Exception as e:
            logger.error(f"执行 Langchain OpenAILLM 测试时发生意外错误: {e}", exc_info=True)

        logger.info("Langchain OpenAILLM __main__ 测试结束。")
    pass

# src/llms/openai_llm.py
# OpenAI LLM 接口实现
import logging
from typing import Any, Dict, List, Optional
import openai  # 使用官方 OpenAI 库

from .base_llm import BaseLLM
from configs.config import OPENAI_API_KEY, DEFAULT_LLM_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, load_config

# 确保在导入或使用此模块时加载配置。
load_config()

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    OpenAI 语言模型 (GPT-3, GPT-3.5-turbo, GPT-4 等) 的封装。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_key: Optional[str] = OPENAI_API_KEY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any
    ):
        """
        初始化 OpenAI LLM。

        参数:
            model_name (str): 要使用的 OpenAI 模型名称 (例如 "gpt-3.5-turbo", "text-davinci-003")。
            api_key (Optional[str]): OpenAI API 密钥。如果为 None，则尝试使用配置中的 OPENAI_API_KEY。
            max_tokens (int): 生成时默认的最大 token 数。
            temperature (float): 默认的采样温度。
            **kwargs: OpenAI 客户端或模型的其他参数。
        """
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            logger.error("OpenAI API 密钥未设置。请提供密钥或在 .env 文件中设置 OPENAI_API_KEY。")
            raise ValueError("OpenAI API 密钥缺失。")

    def _initialize_client(self) -> None:
        """
        初始化 OpenAI API 客户端。
        新的 OpenAI 库 (v1.0+) 使用客户端实例。
        """
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI 客户端已为模型 {self.model_name} 初始化。")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {e}", exc_info=True)
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
        使用非聊天模型 (例如 "text-davinci-003") 生成文本补全。
        对于像 "gpt-3.5-turbo" 这样的聊天模型，请使用 `chat` 方法。
        """
        if self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4"):
            logger.warning(f"模型 {self.model_name} 是一个聊天模型。正在使用 `chat` 方法进行生成。")
            # 适应聊天格式
            messages = [{"role": "user", "content": prompt}]
            chat_response = self.chat(messages, max_tokens, temperature, stop_sequences, **kwargs)
            return chat_response.get("content", "")

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        try:
            logger.debug(
                f"正在使用 OpenAI 模型 {self.model_name} 生成文本。提示: '{prompt[:100]}...'。"
                f"最大 token 数: {current_max_tokens}, 温度: {current_temperature}"
            )
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,  # 传递其他初始化时配置
                **kwargs       # 传递运行时配置
            )
            generated_text = response.choices[0].text.strip()
            logger.debug(f"OpenAI 响应: '{generated_text[:100]}...'")
            return generated_text
        except openai.APIError as e:
            logger.error(f"生成过程中发生 OpenAI API 错误: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"OpenAI 生成过程中发生意外错误: {e}", exc_info=True)
            raise

    async def agenerate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        异步生成文本补全。
        """
        if self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4"):
            logger.warning(f"异步: 模型 {self.model_name} 是一个聊天模型。正在使用 `achat` 方法。")
            messages = [{"role": "user", "content": prompt}]
            chat_response = await self.achat(messages, max_tokens, temperature, stop_sequences, **kwargs)
            return chat_response.get("content", "")

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        # 异步客户端如果尚未初始化，则需要初始化 (或使用 openai.AsyncOpenAI)
        async_client = openai.AsyncOpenAI(api_key=self.api_key)

        try:
            logger.debug(
                f"正在使用 OpenAI 模型 {self.model_name} 异步生成文本。提示: '{prompt[:100]}...'。"
                f"最大 token 数: {current_max_tokens}, 温度: {current_temperature}"
            )
            response = await async_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            generated_text = response.choices[0].text.strip()
            logger.debug(f"异步 OpenAI 响应: '{generated_text[:100]}...'")
            return generated_text
        except openai.APIError as e:
            logger.error(f"异步生成过程中发生 OpenAI API 错误: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"异步 OpenAI 生成过程中发生意外错误: {e}", exc_info=True)
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        使用聊天模型 (例如 "gpt-3.5-turbo", "gpt-4") 生成聊天补全。
        """
        if not (self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4") or "turbo" in self.model_name or "gpt-4o" in self.model_name):
            logger.warning(
                f"模型 {self.model_name} 可能不是聊天模型。"
                "`chat` 方法专为基于聊天的模型设计。"
                "对于补全模型，请考虑使用 `generate`。"
            )

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        try:
            logger.debug(
                f"正在使用 OpenAI 模型 {self.model_name} 生成聊天补全。"
                f"消息: {messages}。最大 token 数: {current_max_tokens}, 温度: {current_temperature}"
            )
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            # 聊天补全的响应结构不同
            # response.choices[0].message 是一个 ChatCompletionMessage 对象
            # 它具有 .role 和 .content 等属性
            response_message = response.choices[0].message
            logger.debug(f"OpenAI 聊天响应: 角色: {response_message.role}, 内容: '{str(response_message.content)[:100]}...'")
            return {"role": response_message.role, "content": response_message.content}
        except openai.APIError as e:
            logger.error(f"聊天补全过程中发生 OpenAI API 错误: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"OpenAI 聊天补全过程中发生意外错误: {e}", exc_info=True)
            raise

    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        异步生成聊天补全。
        """
        if not (self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4") or "turbo" in self.model_name or "gpt-4o" in self.model_name):
            logger.warning(
                f"异步: 模型 {self.model_name} 可能不是聊天模型。"
                "`achat` 方法专为基于聊天的模型设计。"
            )

        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        current_temperature = temperature if temperature is not None else self.temperature

        async_client = openai.AsyncOpenAI(api_key=self.api_key)

        try:
            logger.debug(
                f"正在使用 OpenAI 模型 {self.model_name} 异步生成聊天补全。"
                f"消息: {messages}。最大 token 数: {current_max_tokens}, 温度: {current_temperature}"
            )
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop_sequences,
                **self.config,
                **kwargs
            )
            response_message = response.choices[0].message
            logger.debug(f"异步 OpenAI 聊天响应: 角色: {response_message.role}, 内容: '{str(response_message.content)[:100]}...'")
            return {"role": response_message.role, "content": response_message.content}
        except openai.APIError as e:
            logger.error(f"异步聊天补全过程中发生 OpenAI API 错误: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"异步 OpenAI 聊天补全过程中发生意外错误: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    # 这部分用于测试 OpenAILLM 类
    # 确保在 .env 文件中设置了 OPENAI_API_KEY
    from configs.logging_config import setup_logging
    # 配置已在模块顶部的 load_config() 调用中加载。
    setup_logging()

    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        logger.warning("OPENAI_API_KEY 未设置或为占位符。正在跳过 OpenAILLM 测试。")
    else:
        logger.info("正在测试 OpenAILLM...")

        # 使用聊天模型进行测试
        try:
            chat_llm = OpenAILLM(model_name="gpt-3.5-turbo") # 或者 DEFAULT_LLM_MODEL 如果它是聊天模型
            logger.info(f"聊天 LLM 信息: {chat_llm.get_model_info()}")

            chat_messages = [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": "法国的首都是哪里？"}
            ]
            chat_response = chat_llm.chat(chat_messages)
            logger.info(f"对于“法国的首都是哪里？”的聊天响应: {chat_response['content']}")
            assert "paris" in chat_response['content'].lower() # 预期答案包含 "paris"

            # 使用聊天模型测试 generate (应该能自适应)
            # generate_response_chat_model = chat_llm.generate("把 'hello' 翻译成法语。")
            # logger.info(f"对于“把 'hello' 翻译成法语”的 generate 响应 (来自聊天模型): {generate_response_chat_model}")
            # assert "bonjour" in generate_response_chat_model.lower()

        except Exception as e:
            logger.error(f"OpenAILLM 聊天模型测试期间出错: {e}", exc_info=True)

        # 使用补全模型进行测试 (例如 gpt-3.5-turbo-instruct，如果你有权限或可用)
        # 注意: "text-davinci-003" 已弃用。"gpt-3.5-turbo-instruct" 是更新的 instruct 模型。
        # 如果你没有 instruct 模型的权限，这部分可能会失败或需要调整。
        # 目前，我们假设用户可能不容易访问非聊天补全模型。
        # logger.info("正在跳过非聊天补全模型测试，因为它们现在不太常见。")
        # try:
        #     instruct_model_name = "gpt-3.5-turbo-instruct" # 或其他补全模型
        #     completion_llm = OpenAILLM(model_name=instruct_model_name)
        #     logger.info(f"补全 LLM 信息: {completion_llm.get_model_info()}")
        #     completion_prompt = "从前，" # 中文示例："很久很久以前，"
        #     completion_response = completion_llm.generate(completion_prompt, max_tokens=50)
        #     logger.info(f"对于“从前，”的补全响应: {completion_response}")
        #     assert len(completion_response) > 5
        # except openai.NotFoundError as e:
        #    logger.warning(f"无法测试 instruct 模型 {instruct_model_name}，你的 API 密钥可能无法访问它: {e}")
        # except Exception as e:
        #    logger.error(f"OpenAILLM 补全模型测试期间出错: {e}", exc_info=True)


        # 异步测试 (可选，可以使用事件循环单独运行)
        import asyncio
        async def run_async_tests():
            logger.info("正在运行异步测试...")
            try:
                async_chat_llm = OpenAILLM(model_name="gpt-3.5-turbo")
                async_chat_messages = [
                    {"role": "user", "content": "异步编程的三个好处是什么？"}
                ]
                async_response = await async_chat_llm.achat(async_chat_messages, max_tokens=150)
                logger.info(f"异步聊天响应: {async_response['content']}")
                assert async_response['content']

                # 使用聊天模型进行异步 generate
                # async_gen_response = await async_chat_llm.agenerate("给我讲个短笑话。")
                # logger.info(f"异步 generate 响应 (来自聊天模型): {async_gen_response}")
                # assert async_gen_response

            except Exception as e:
                logger.error(f"异步 OpenAILLM 测试期间出错: {e}", exc_info=True)

        logger.info("正在尝试运行异步测试...")
        try:
            asyncio.run(run_async_tests())
        except RuntimeError as e:
            # 如果事件循环已在运行 (例如在 Jupyter 中)，则可能发生这种情况
            logger.warning(f"无法直接运行 asyncio.run (可能是由于现有事件循环): {e}。在此环境中，你可能需要以不同方式运行异步测试。")
        except Exception as e:
            logger.error(f"运行异步测试时发生一般错误: {e}", exc_info=True)

        logger.info("OpenAILLM 测试完成。")

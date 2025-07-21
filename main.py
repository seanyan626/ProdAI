# main.py
"""
AI 项目框架主入口文件

提供模块化的测试功能，支持对各个骨架模块进行独立测试验证。
使用方法：
1. 在 .env 文件中配置相应的 API 密钥
2. 运行 python main.py --test <module_name> 来测试特定模块
3. 或直接运行 python main.py 进行交互式选择
"""

import argparse
import logging

from configs.config import config
from configs.logging_config import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


class ModuleTester:
    """模块测试器类，负责组织和执行各个模块的测试"""

    def __init__(self):
        self.test_modules = {
            "langchain_openai_llm": self.test_langchain_openai_llm,
            "langchain_dashscope_llm": self.test_langchain_dashscope_llm,
            "langchain_deepseek_llm": self.test_langchain_deepseek_llm,
            "llamaindex_openai_llm": self.test_llamaindex_openai_llm,
            "llamaindex_dashscope_llm": self.test_llamaindex_dashscope_llm,
            "llamaindex_deepseek_llm": self.test_llamaindex_deepseek_llm,
            "langchain_openai_embedding": self.test_langchain_openai_embedding,
            "langchain_dashscope_embedding": self.test_langchain_dashscope_embedding,
            "llamaindex_openai_embedding": self.test_llamaindex_openai_embedding,
            "llamaindex_dashscope_embedding": self.test_llamaindex_dashscope_embedding,
            "dashscope_ocr": self.test_dashscope_ocr,
            "ocr_models": self.test_ocr_models,
            "agents": self.test_agents,
            "rag": self.test_rag,
            "tools": self.test_tools,
            "memory": self.test_memory,
        }

    def test_langchain_openai_llm(self):
        """测试 LangChain OpenAI LLM 模型"""
        logger.info("=== 测试 LangChain OpenAI LLM 模型 ===")

        if config.openai.api_key:
            try:
                from core.models.llm.langchain.openai_llm import OpenAILLM
                logger.info("测试 OpenAI LLM...")

                llm = OpenAILLM()
                messages = [{"role": "user", "content": "Hello, introduce yourself briefly."}]
                response = llm.chat(messages)
                logger.info(f"✅ OpenAI LLM 响应: {response.get('content', '')[:100]}...")

            except Exception as e:
                logger.error(f"❌ OpenAI LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 OpenAI API 密钥，跳过测试")

    def test_langchain_dashscope_llm(self):
        """测试 LangChain DashScope LLM 模型"""
        logger.info("=== 测试 LangChain DashScope LLM 模型 ===")

        if config.dashscope.api_key:
            try:
                from core.models.llm.langchain.dashscope_llm import DashScopeLLM
                logger.info("测试 DashScope LLM...")

                llm = DashScopeLLM()
                messages = [{"role": "user", "content": "你好，简单介绍一下你自己。"}]
                response = llm.chat(messages)
                logger.info(f"✅ DashScope LLM 响应: {response.get('content', '')[:100]}...")

            except Exception as e:
                logger.error(f"❌ DashScope LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DashScope API 密钥，跳过测试")

    def test_langchain_deepseek_llm(self):
        """测试 LangChain DeepSeek LLM 模型"""
        logger.info("=== 测试 LangChain DeepSeek LLM 模型 ===")

        if config.deepseek.api_key:
            try:
                from core.models.llm.langchain.deepseek_llm import DeepSeekLLM
                logger.info("测试 De2epSeek LLM...")

                llm = DeepSeekLLM()
                messages = [{"role": "user", "content": "你好，简单介绍一下你自己。"}]
                response = llm.chat(messages)
                logger.info(f"✅ DeepSeek LLM 响应: {response.get('content', '')[:100]}...")

            except Exception as e:
                logger.error(f"❌ DeepSeek LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DeepSeek API 密钥，跳过测试")

    def test_llamaindex_openai_llm(self):
        """测试 LlamaIndex OpenAI LLM 模型"""
        logger.info("=== 测试 LlamaIndex OpenAI LLM 模型 ===")

        try:
            from llama_index.core.llms import ChatMessage
        except ImportError:
            logger.error("❌ 无法导入 LlamaIndex，请安装相关依赖")
            return

        if config.openai.api_key:
            try:
                from core.models.llm.llama_index.openai_llm import OpenAILLM
                logger.info("测试 OpenAI LlamaIndex LLM...")

                llm = OpenAILLM()
                messages = [ChatMessage(role="user", content="Hello, introduce yourself briefly.")]
                response = llm.chat(messages)
                logger.info(f"✅ OpenAI LlamaIndex LLM 响应: {response.message.content[:100]}...")

            except Exception as e:
                logger.error(f"❌ OpenAI LlamaIndex LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 OpenAI API 密钥，跳过测试")

    def test_llamaindex_dashscope_llm(self):
        """测试 LlamaIndex DashScope LLM 模型"""
        logger.info("=== 测试 LlamaIndex DashScope LLM 模型 ===")

        try:
            from llama_index.core.llms import ChatMessage
        except ImportError:
            logger.error("❌ 无法导入 LlamaIndex，请安装相关依赖")
            return

        if config.dashscope.api_key:
            try:
                from core.models.llm.llama_index.dashscope_llm import DashScopeLLM
                logger.info("测试 DashScope LlamaIndex LLM...")

                llm = DashScopeLLM()
                messages = [ChatMessage(role="user", content="你好，简单介绍一下你自己。")]
                response = llm.chat(messages)
                logger.info(f"✅ DashScope LlamaIndex LLM 响应: {response.message.content[:100]}...")

            except Exception as e:
                logger.error(f"❌ DashScope LlamaIndex LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DashScope API 密钥，跳过测试")

    def test_llamaindex_deepseek_llm(self):
        """测试 LlamaIndex DeepSeek LLM 模型"""
        logger.info("=== 测试 LlamaIndex DeepSeek LLM 模型 ===")

        try:
            from llama_index.core.llms import ChatMessage
        except ImportError:
            logger.error("❌ 无法导入 LlamaIndex，请安装相关依赖")
            return

        if config.deepseek.api_key:
            try:
                from core.models.llm.llama_index.deepseek_llm import DeepSeekLLM
                logger.info("测试 DeepSeek LlamaIndex LLM...")

                llm = DeepSeekLLM()
                messages = [ChatMessage(role="user", content="你好，简单介绍一下你自己。")]
                response = llm.chat(messages)
                logger.info(f"✅ DeepSeek LlamaIndex LLM 响应: {response.message.content[:100]}...")

            except Exception as e:
                logger.error(f"❌ DeepSeek LlamaIndex LLM 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DeepSeek API 密钥，跳过测试")

    def test_langchain_openai_embedding(self):
        """测试 LangChain OpenAI Embedding 模型"""
        logger.info("=== 测试 LangChain OpenAI Embedding 模型 ===")

        if config.openai.api_key:
            try:
                from core.models.embedding.langchain.openai_embedding import OpenAIEmbedding
                logger.info("测试 OpenAI Embedding...")

                embedding = OpenAIEmbedding()
                text = "Hello, this is a test."
                vector = embedding.embed_query(text)
                logger.info(f"✅ OpenAI Embedding 向量维度: {len(vector)}, 前5维: {vector[:5]}")

            except Exception as e:
                logger.error(f"❌ OpenAI Embedding 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 OpenAI API 密钥，跳过测试")

    def test_langchain_dashscope_embedding(self):
        """测试 LangChain DashScope Embedding 模型"""
        logger.info("=== 测试 LangChain DashScope Embedding 模型 ===")

        if config.dashscope.api_key:
            try:
                from core.models.embedding.langchain.dashscope_embedding import DashScopeEmbedding
                logger.info("测试 DashScope Embedding...")

                embedding = DashScopeEmbedding()
                text = "你好，这是一个测试。"
                vector = embedding.embed_query(text)
                logger.info(f"✅ DashScope Embedding 向量维度: {len(vector)}, 前5维: {vector[:5]}")

            except Exception as e:
                logger.error(f"❌ DashScope Embedding 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DashScope API 密钥，跳过测试")

    def test_llamaindex_openai_embedding(self):
        """测试 LlamaIndex OpenAI Embedding 模型"""
        logger.info("=== 测试 LlamaIndex OpenAI Embedding 模型 ===")

        if config.openai.api_key:
            try:
                from core.models.embedding.llama_index.openai_embedding import OpenAIEmbedding
                logger.info("测试 OpenAI LlamaIndex Embedding...")

                embedding = OpenAIEmbedding()
                text = "Hello, this is a test."
                vector = embedding.get_text_embedding(text)
                logger.info(f"✅ OpenAI LlamaIndex Embedding 向量维度: {len(vector)}, 前5维: {vector[:5]}")

            except Exception as e:
                logger.error(f"❌ OpenAI LlamaIndex Embedding 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 OpenAI API 密钥，跳过测试")

    def test_llamaindex_dashscope_embedding(self):
        """测试 LlamaIndex DashScope Embedding 模型"""
        logger.info("=== 测试 LlamaIndex DashScope Embedding 模型 ===")

        if config.dashscope.api_key:
            try:
                from core.models.embedding.llama_index.dashscope_embedding import DashScopeEmbedding
                logger.info("测试 DashScope LlamaIndex Embedding...")

                embedding = DashScopeEmbedding()
                text = "你好，这是一个测试。"
                vector = embedding.get_text_embedding(text)
                logger.info(f"✅ DashScope LlamaIndex Embedding 向量维度: {len(vector)}, 前5维: {vector[:5]}")

            except Exception as e:
                logger.error(f"❌ DashScope LlamaIndex Embedding 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置 DashScope API 密钥，跳过测试")

    def test_dashscope_ocr(self):
        """测试DashScope OCR 模型"""
        logger.info("=== 测试DashScope OCR 模型 ===")

        if config.dashscope.api_key:
            try:
                from core.models.ocr.cloud.dashscope_ocr_model import DashScopeOCRModel
                import os
                logger.info("测试DashScope OCR...")

                # 创建DashScope OCR 实例
                ocr = DashScopeOCRModel()
                logger.info(f"✅ DashScope OCR 模型信息: {ocr.get_model_info()}")

                # 查找测试图像文件
                test_image_paths = [
                    "data/images/1.png",
                    "data/images/test_image.jpg",
                    "data/images/test_image.png",
                    "data/images/test.jpg",
                    "data/images/test.png",
                    "data/images/sample.jpg",
                    "data/images/sample.png",
                    "data/images/image.jpg",
                    "data/images/image.png"
                ]

                test_image = None
                for image_path in test_image_paths:
                    if os.path.exists(image_path):
                        test_image = image_path
                        break

                if test_image:
                    logger.info(f"📷 找到测试图像: {test_image}")
                    logger.info("🔍 开始进行OCR识别...")

                    # 进行实际的OCR识别
                    result = ocr.recognize_text(test_image)

                    logger.info(f"✅ OCR识别完成!")
                    logger.info(f"📝 识别结果: {result.full_text}")
                    logger.info(f"🎯 置信度: {result.confidence:.2f}")
                    logger.info(f"📊 文本段数量: {len(result.segments)}")

                    # 显示前3个文本段的详细信息
                    for i, segment in enumerate(result.segments[:3]):
                        logger.info(f"   段落 {i + 1}: '{segment['text']}' (置信度: {segment['confidence']:.2f})")

                else:
                    logger.info("✅ DashScope OCR 初始化成功")
                    logger.info("💡 提示：在 data/images 目录下放置测试图像文件可进行实际OCR测试")
                    logger.info("📁 支持的文件名: 1.png, test_image.jpg, test.jpg, sample.jpg 等")
                    logger.info("📂 当前已有文件: data/images/1.png")

            except Exception as e:
                logger.error(f"❌ DashScope OCR 测试失败: {e}")
        else:
            logger.warning("⚠️ 未配置DashScope API 密钥，跳过测试")
            logger.info("💡 请在 .env 文件中设置 DASHSCOPE_API_KEY")
            logger.info("🔑 API密钥应该是您的 DashScope API Key")

    def test_ocr_models(self):
        """测试 OCR 模型"""
        logger.info("=== 测试 OCR 模型 ===")

        # 测试本地 OCR 模型
        try:
            from core.models.ocr.local.easy_ocr_model import EasyOCRModel, EASYOCR_AVAILABLE
            if EASYOCR_AVAILABLE:
                logger.info("测试 EasyOCR 模型...")
                ocr = EasyOCRModel(languages=['en'], gpu=False)
                logger.info(f"✅ EasyOCR 模型信息: {ocr.get_model_info()}")
            else:
                logger.warning("⚠️ EasyOCR 库未安装，跳过测试")
        except Exception as e:
            logger.error(f"❌ EasyOCR 测试失败: {e}")

        try:
            from core.models.ocr.local.paddle_ocr_model import PaddleOCRModel, PADDLEOCR_AVAILABLE
            if PADDLEOCR_AVAILABLE:
                logger.info("测试 PaddleOCR 模型...")
                ocr = PaddleOCRModel(lang='ch', use_gpu=False)
                logger.info(f"✅ PaddleOCR 模型信息: {ocr.get_model_info()}")
            else:
                logger.warning("⚠️ PaddleOCR 库未安装，跳过测试")
        except Exception as e:
            logger.error(f"❌ PaddleOCR 测试失败: {e}")

    def test_agents(self):
        """测试 Agent 模块"""
        logger.info("=== 测试 Agent 模块 ===")

        try:
            from core.agents.base_agent import BaseAgent
            logger.info("✅ BaseAgent 导入成功")

            from core.agents.specific_agent import SpecificAgent
            logger.info("✅ SpecificAgent 导入成功")

        except Exception as e:
            logger.error(f"❌ Agent 模块测试失败: {e}")

    def test_rag(self):
        """测试 RAG 模块"""
        logger.info("=== 测试 RAG 模块 ===")

        try:
            from core.rag.base_retriever import BaseRetriever
            logger.info("✅ BaseRetriever 导入成功")

            from core.rag.simple_rag import SimpleRAG
            logger.info("✅ SimpleRAG 导入成功")

        except Exception as e:
            logger.error(f"❌ RAG 模块测试失败: {e}")

    def test_tools(self):
        """测试 Tools 模块"""
        logger.info("=== 测试 Tools 模块 ===")

        try:
            from core.tools.base_tool import BaseTool
            logger.info("✅ BaseTool 导入成功")

            from core.tools.search_tool import SearchTool
            logger.info("✅ SearchTool 导入成功")

        except Exception as e:
            logger.error(f"❌ Tools 模块测试失败: {e}")

    def test_memory(self):
        """测试 Memory 模块"""
        logger.info("=== 测试 Memory 模块 ===")

        try:
            from core.memory.base_memory import BaseMemory
            logger.info("✅ BaseMemory 导入成功")

            from core.memory.simple_memory import SimpleMemory
            logger.info("✅ SimpleMemory 导入成功")

        except Exception as e:
            logger.error(f"❌ Memory 模块测试失败: {e}")

    def run_test(self, module_name: str):
        """运行指定模块的测试"""
        if module_name not in self.test_modules:
            logger.error(f"❌ 未知的测试模块: {module_name}")
            logger.info(f"可用的测试模块: {', '.join(self.test_modules.keys())}")
            return

        logger.info(f"开始测试模块: {module_name}")
        try:
            self.test_modules[module_name]()
            logger.info(f"✅ 模块 {module_name} 测试完成")
        except Exception as e:
            logger.error(f"❌ 模块 {module_name} 测试失败: {e}")

    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行所有模块测试...")
        for module_name in self.test_modules:
            self.run_test(module_name)
            logger.info("-" * 50)

    def interactive_mode(self):
        """交互式模式"""
        logger.info(f"欢迎使用 {config.app.name} 测试工具")
        logger.info("可用的测试模块:")

        for i, module_name in enumerate(self.test_modules.keys(), 1):
            print(f"  {i}. {module_name}")

        print(f"  {len(self.test_modules) + 1}. 运行所有测试")
        print("  0. 退出")

        while True:
            try:
                choice = input("\n请选择要测试的模块 (输入数字): ").strip()

                if choice == "0":
                    logger.info("退出测试工具")
                    break

                choice_num = int(choice)
                module_names = list(self.test_modules.keys())

                if choice_num == len(self.test_modules) + 1:
                    self.run_all_tests()
                elif 1 <= choice_num <= len(self.test_modules):
                    module_name = module_names[choice_num - 1]
                    self.run_test(module_name)
                else:
                    print("❌ 无效的选择，请重新输入")

            except (ValueError, KeyboardInterrupt):
                logger.info("退出测试工具")
                break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI 项目框架测试工具")
    parser.add_argument(
        "--test",
        type=str,
        help="指定要测试的模块名称"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有测试"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的测试模块"
    )

    args = parser.parse_args()
    tester = ModuleTester()

    if args.list:
        print("可用的测试模块:")
        for module_name in tester.test_modules.keys():
            print(f"  - {module_name}")
        return

    if args.all:
        tester.run_all_tests()
    elif args.test:
        tester.run_test(args.test)
    else:
        tester.interactive_mode()


if __name__ == "__main__":
    main()

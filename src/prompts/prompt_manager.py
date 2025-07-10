# src/prompts/prompt_manager.py
# 提示管理器模块
import logging
import os
from typing import Dict, Any, Optional, List
from string import Template # 使用 string.Template 进行简单的变量替换

# 假设你的项目结构是 aiproject/src/prompts/templates
# 并且此文件是 aiproject/src/prompts/prompt_manager.py
DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates") # 默认模板目录

logger = logging.getLogger(__name__)

class PromptTemplate(Template):
    """
    自定义模板类，如果需要，可以允许使用不同的分隔符，
    并且将来可能添加更复杂的模板逻辑。
    目前，它是 string.Template 的直接子类。
    """
    # delimiter = "$" # 默认分隔符，必要时可以更改
    pass


class PromptManager:
    """
    管理提示模板的加载、格式化和访问。
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化 PromptManager。

        参数:
            templates_dir (Optional[str]): 存储提示模板文件的目录。
                                           默认为此文件旁边的 "templates" 子目录。
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.loaded_templates: Dict[str, PromptTemplate] = {} # 已加载的模板
        self._load_all_templates()
        logger.info(f"PromptManager 已初始化。已从以下位置加载模板: {self.templates_dir}")

    def _load_template_file(self, file_path: str) -> Optional[str]:
        """加载单个模板文件。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            logger.error(f"模板文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"加载模板文件 {file_path} 出错: {e}", exc_info=True)
            return None

    def _load_all_templates(self) -> None:
        """
        从模板目录加载所有 .txt 或 .prompt 文件。
        文件名 (不带扩展名) 用作模板名称。
        """
        if not os.path.isdir(self.templates_dir):
            logger.warning(f"模板目录未找到: {self.templates_dir}。未加载任何模板。")
            return

        for filename in os.listdir(self.templates_dir):
            if filename.endswith((".txt", ".prompt")): # 支持的模板文件扩展名
                template_name = os.path.splitext(filename)[0] # 使用文件名作为模板名
                file_path = os.path.join(self.templates_dir, filename)
                content = self._load_template_file(file_path)
                if content:
                    self.loaded_templates[template_name] = PromptTemplate(content)
                    logger.debug(f"已加载模板: '{template_name}' (来自 {filename})")
        logger.info(f"已加载 {len(self.loaded_templates)} 个模板。")


    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        检索已加载的提示模板。

        参数:
            template_name (str): 模板的名称 (不带扩展名的文件名)。

        返回:
            Optional[PromptTemplate]: PromptTemplate 对象，如果未找到则为 None。
        """
        template = self.loaded_templates.get(template_name)
        if not template:
            logger.warning(f"模板 '{template_name}' 未找到。")
        return template

    def format_prompt(self, template_name: str, **kwargs: Any) -> Optional[str]:
        """
        使用命名模板和提供的关键字参数格式化提示。

        参数:
            template_name (str): 要使用的模板的名称。
            **kwargs: 要替换到模板中的变量。

        返回:
            Optional[str]: 格式化的提示字符串，如果找不到模板或格式化过程中出错，则为 None。
        """
        template = self.get_template(template_name)
        if not template:
            return None

        try:
            # 使用 safe_substitute 以避免在某些占位符丢失时出现 KeyError，
            # 并允许非用于替换的 $ 符号。
            # 如果需要严格替换，请使用 substitute()。
            formatted_prompt = template.safe_substitute(**kwargs)
            logger.debug(f"已使用变量 {kwargs} 格式化提示 '{template_name}'。")
            return formatted_prompt
        except KeyError as e:
            logger.error(f"模板 '{template_name}' 缺少变量: {e}。提供的变量: {kwargs}")
            return None
        except Exception as e:
            logger.error(f"格式化模板 '{template_name}' 出错: {e}", exc_info=True)
            return None

    def list_available_templates(self) -> List[str]:
        """
        返回所有已加载模板的名称列表。
        """
        return list(self.loaded_templates.keys())

# --- 示例: 创建用于测试的虚拟模板文件 ---
def _create_dummy_template_files(templates_dir: str):
    """为测试创建虚拟模板文件。"""
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # 模板 1: simple_greeting.txt (简单问候)
    with open(os.path.join(templates_dir, "simple_greeting.txt"), "w", encoding="utf-8") as f:
        f.write("你好，$name！欢迎来到 $platform。")

    # 模板 2: question_template.prompt (问题模板)
    with open(os.path.join(templates_dir, "question_template.prompt"), "w", encoding="utf-8") as f:
        f.write("用户查询: $query\n上下文: $context\n\n请根据上下文回答查询。")

    # 模板 3: escape_test.txt (用于 safe_substitute 测试的模板)
    with open(os.path.join(templates_dir, "escape_test.txt"), "w", encoding="utf-8") as f:
        f.write("此模板花费 $$10。你的变量是 $var。")


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    load_config()
    setup_logging()

    logger.info("正在测试 PromptManager...")

    # 为测试创建虚拟模板
    test_templates_path = os.path.join(os.path.dirname(__file__), "test_prompt_templates_示例") # 测试模板目录
    _create_dummy_template_files(test_templates_path)

    manager = PromptManager(templates_dir=test_templates_path)

    logger.info(f"可用模板: {manager.list_available_templates()}")
    assert "simple_greeting" in manager.list_available_templates()
    assert "question_template" in manager.list_available_templates()

    # 测试格式化有效提示
    greeting = manager.format_prompt("simple_greeting", name="爱丽丝", platform="AI 世界")
    logger.info(f"格式化的 'simple_greeting': {greeting}")
    assert greeting == "你好，爱丽丝！欢迎来到 AI 世界。"

    # 测试格式化另一个有效提示
    question = manager.format_prompt(
        "question_template",
        query="法国的首都是哪里？",
        context="法国是欧洲的一个国家。其首都是巴黎。"
    )
    logger.info(f"格式化的 'question_template': {question}")
    assert "用户查询: 法国的首都是哪里？" in question
    assert "上下文: 法国是欧洲的一个国家。其首都是巴黎。" in question

    # 测试使用 safe_substitute 格式化带缺失变量的模板
    escaped_prompt = manager.format_prompt("escape_test", var="测试值")
    logger.info(f"格式化的 'escape_test': {escaped_prompt}")
    assert escaped_prompt == "此模板花费 $10。你的变量是 测试值。" # `$$` 会转义为 `$`

    # 测试格式化模板时所有变量均缺失的情况
    missing_vars_prompt = manager.format_prompt("simple_greeting")
    logger.info(f"格式化的 'simple_greeting' (无变量): {missing_vars_prompt}")
    assert missing_vars_prompt == "你好，$name！欢迎来到 $platform。"


    # 测试获取不存在的模板
    non_existent = manager.format_prompt("non_existent_template_不存在的模板", key="值")
    assert non_existent is None

    # 清理虚拟模板
    import shutil
    try:
        shutil.rmtree(test_templates_path)
        logger.info(f"已清理测试模板目录: {test_templates_path}")
    except Exception as e:
        logger.error(f"清理测试模板出错: {e}")

    logger.info("PromptManager 测试成功完成。")

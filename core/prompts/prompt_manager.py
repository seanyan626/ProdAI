# core/prompts/prompt_manager.py
# 提示管理器模块
import logging
import os
from string import Template
from typing import Dict, Any, Optional, List

DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")  # 默认模板目录

logger = logging.getLogger(__name__)


class PromptTemplate(Template):
    """
    自定义模板类，如果需要，可以允许使用不同的分隔符，
    并且将来可能添加更复杂的模板逻辑。
    目前，它是 string.Template 的直接子类。
    """
    pass


class PromptManager:
    """
    管理提示模板的加载、格式化和访问。
    （实现待补充）
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        初始化 PromptManager。

        参数:
            templates_dir (Optional[str]): 存储提示模板文件的目录。
                                           默认为此文件旁边的 "templates" 子目录。
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.loaded_templates: Dict[str, PromptTemplate] = {}  # 已加载的模板
        # self._load_all_templates() # 实际加载逻辑待补充
        logger.info(f"PromptManager 已初始化。模板目录: {self.templates_dir}")

    def _load_template_file(self, file_path: str) -> Optional[str]:
        """
        加载单个模板文件。
        （实现待补充）
        """
        pass

    def _load_all_templates(self) -> None:
        """
        从模板目录加载所有 .txt 或 .prompt 文件。
        文件名 (不带扩展名) 用作模板名称。
        （实现待补充）
        """
        pass

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        检索已加载的提示模板。
        （实现待补充）
        """
        # template = self.loaded_templates.get(template_name)
        # if not template:
        #     logger.warning(f"模板 '{template_name}' 未找到。")
        # return template
        return None  # 占位符

    def format_prompt(self, template_name: str, **kwargs: Any) -> Optional[str]:
        """
        使用命名模板和提供的关键字参数格式化提示。
        （实现待补充）
        """
        # template = self.get_template(template_name)
        # if not template:
        #     return None
        # try:
        #     formatted_prompt = template.safe_substitute(**kwargs)
        #     logger.debug(f"已使用变量 {kwargs} 格式化提示 '{template_name}'。")
        #     return formatted_prompt
        # except KeyError as e:
        #     logger.error(f"模板 '{template_name}' 缺少变量: {e}。提供的变量: {kwargs}")
        #     return None
        # except Exception as e:
        #     logger.error(f"格式化模板 '{template_name}' 出错: {e}", exc_info=True)
        #     return None
        return None  # 占位符

    def list_available_templates(self) -> List[str]:
        """
        返回所有已加载模板的名称列表。
        （实现待补充）
        """
        return []  # 占位符


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging

    load_config()
    setup_logging()
    logger.info("PromptManager 模块可以直接运行测试（如果包含测试代码）。")
    # 此处可以添加直接测试此模块内函数的代码
    pass

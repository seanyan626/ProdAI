# 此文件使 'core' 目录成为一个 Python 包。

# 你可以在这里为你的源代码包定义一个版本，例如：
__version__ = "0.1.0"  # 版本号

# 你也可以选择直接暴露子模块的某些部分，
# 不过这通常在子模块自己的 __init__.py 文件中完成。
# 示例 (如果你想执行 `from core import agents` 然后 `agents.BaseAgent`):
# from . import agents
# from . import models # 现在模型都在 models 包下
# from . import memory
# from . import prompts
# from . import rag
# from . import tools
# from . import utils

# 或者，如果你想直接在 core 级别暴露特定的类/函数 (对于大型项目不太常见):
# from .agents.base_agent import BaseAgent
# from .models.llm.base_llm import BaseLLM # 更新的路径和类名

# src/tools/search_tool.py
# 示例搜索工具实现
import logging
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, Field
import requests # 示例外部依赖，请将 'requests' 添加到 requirements.txt

from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class SearchToolInput(BaseModel):
    query: str = Field(..., description="搜索查询字符串。")
    num_results: int = Field(3, description="要返回的搜索结果数量。", ge=1, le=10) # ge=大于等于, le=小于等于

# 这是一个模拟的搜索函数。在实际场景中，
# 你需要与搜索引擎 API (Google, Bing, DuckDuckGo, Tavily 等) 集成。
def mock_search_api(query: str, num_results: int) -> Dict[str, Any]:
    """
    模拟对外部搜索 API 的调用。
    """
    logger.info(f"模拟搜索 API 被调用，查询: '{query}', 结果数量: {num_results}")
    results = []
    for i in range(num_results):
        results.append({
            "title": f"关于 '{query}' 的模拟搜索结果 {i+1}",
            "link": f"https://example.com/search?q={query.replace(' ', '+')}&page={i+1}",
            "snippet": f"这是与 '{query}' 相关的搜索结果 {i+1} 的模拟摘要。"
                       "它包含一些关于在链接处找到的内容的描述性文本。"
        })
    return {"query": query, "results": results}


class SearchTool(BaseTool):
    """
    一个使用 (模拟的) 搜索引擎执行网络搜索的工具。
    """
    name: str = "web_search" # 工具名称：网页搜索
    description: str = (
        "为给定的查询执行网络搜索并返回结果列表。"
        "可用于在互联网上查找信息、时事或特定主题。"
    )
    args_schema: Type[BaseModel] = SearchToolInput

    # 你可以在此处添加配置，例如真实搜索引擎的 API 密钥
    # def __init__(self, api_key: Optional[str] = None, **kwargs):
    #     super().__init__(**kwargs)
    #     self.api_key = api_key
    #     if not self.api_key:
    #         logger.warning("SearchTool 初始化时没有 API 密钥。将使用模拟搜索。")


    def _run(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        执行搜索。

        参数:
            query (str): 搜索查询。
            num_results (int): 要返回的结果数量。

        返回:
            Dict[str, Any]: 包含搜索查询和结果列表的字典，
                            其中每个结果都有标题、链接和摘要。
                            如果搜索失败，则返回错误消息字符串。
        """
        logger.info(f"正在执行 SearchTool，查询: '{query}', 结果数量: {num_results}")
        try:
            # 在真实的工具中，你会使用 self.api_key 并发出请求
            # 例如，使用 'requests' 库:
            # response = requests.get(
            #     "https://api.somesearch_engine.com/v1/search",
            #     params={"q": query, "num": num_results},
            #     headers={"Authorization": f"Bearer {self.api_key}"}
            # )
            # response.raise_for_status() # 对 HTTP 错误引发异常
            # search_results = response.json()

            # 在此示例中使用模拟函数
            search_results = mock_search_api(query, num_results)

            if not search_results or "results" not in search_results or not search_results["results"]:
                return {"query": query, "results": [], "message": "未找到结果。"}

            return search_results

        except requests.exceptions.RequestException as e:
            logger.error(f"SearchTool: 查询 '{query}' 时发生 HTTP 请求错误: {e}", exc_info=True)
            return {"error": f"搜索期间发生网络错误: {e}"}
        except Exception as e:
            logger.error(f"SearchTool: 查询 '{query}' 时发生意外错误: {e}", exc_info=True)
            return {"error": f"搜索期间发生意外错误: {e}"}

    # 异步版本示例 (如果 API 支持)
    async def _arun(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        logger.info(f"正在异步执行 SearchTool，查询: '{query}', 结果数量: {num_results}")
        # 如果使用 httpx 进行异步请求:
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     try:
        #         response = await client.get(
        #             "https://api.somesearch_engine.com/v1/search",
        #             params={"q": query, "num": num_results},
        #             headers={"Authorization": f"Bearer {self.api_key}"}
        #         )
        #         response.raise_for_status()
        #         search_results = response.json()
        #         if not search_results or "results" not in search_results or not search_results["results"]:
        #             return {"query": query, "results": [], "message": "未找到结果。"}
        #         return search_results
        #     except httpx.RequestError as e:
        #         logger.error(f"SearchTool (异步): 查询 '{query}' 时发生 HTTP 请求错误: {e}", exc_info=True)
        #         return {"error": f"异步搜索期间发生网络错误: {e}"}
        #     except Exception as e:
        #         logger.error(f"SearchTool (异步): 查询 '{query}' 时发生意外错误: {e}", exc_info=True)
        #         return {"error": f"异步搜索期间发生意外错误: {e}"}

        # 对于此示例，我们将仅调用模拟的同步版本 (不是真正的异步)
        # 要使其真正异步，mock_search_api 也需要是异步的。
        # import asyncio
        # await asyncio.sleep(0.05) # 模拟异步延迟
        return self._run(query, num_results) # 为模拟回退到同步


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    import asyncio

    load_config()
    setup_logging()

    logger.info("正在测试 SearchTool...")
    search_tool = SearchTool() # 如果使用真实的 API 密钥，请添加 api_key="YOUR_KEY"

    # 使用有效输入进行测试
    results1 = search_tool.run(query="最新AI进展", num_results=2)
    logger.info(f"SearchTool 结果 (最新AI进展, 2): {results1}")
    assert "results" in results1
    assert len(results1["results"]) == 2
    assert "模拟搜索结果 1" in results1["results"][0]["title"]

    # 使用默认结果数量进行测试
    results2 = search_tool.run(query="python编程技巧")
    logger.info(f"SearchTool 结果 (python编程技巧, 默认 3): {results2}")
    assert "results" in results2
    assert len(results2["results"]) == 3

    # 使用 Pydantic 模型输入进行测试
    input_data = SearchToolInput(query="最佳披萨食谱", num_results=1)
    results3 = search_tool.run(input_data)
    logger.info(f"SearchTool 结果 (Pydantic 输入): {results3}")
    assert "results" in results3
    assert len(results3["results"]) == 1
    assert "披萨食谱" in results3["results"][0]["title"] # "pizza recipe" -> "披萨食谱" (取决于模拟API的本地化)
                                                       # 当前模拟API未本地化标题，所以这里可能需要调整或接受英文

    # 测试无效输入 (例如，num_results 超出范围，由 Pydantic 处理)
    invalid_results = search_tool.run(query="测试", num_results=20) # ge=1, le=10
    logger.info(f"SearchTool 结果 (无效的 num_results): {invalid_results}")
    assert "输入验证失败" in str(invalid_results) # 检查中文错误信息

    # 测试异步运行
    async def run_async_search():
        async_results = await search_tool.arun(query="异步搜索测试", num_results=1)
        logger.info(f"SearchTool 异步结果: {async_results}")
        assert "results" in async_results
        assert len(async_results["results"]) == 1
        assert "异步搜索测试" in async_results["results"][0]["title"]

    asyncio.run(run_async_search())

    logger.info(f"SearchTool Schema: {SearchTool.get_tool_info()}")
    logger.info("SearchTool 测试成功完成。")

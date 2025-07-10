# src/tools/search_tool.py
import logging
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, Field
import requests # Example external dependency, add 'requests' to requirements.txt

from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query string.")
    num_results: int = Field(3, description="Number of search results to return.", ge=1, le=10)

# This is a mock search function. In a real scenario,
# you would integrate with a search API (Google, Bing, DuckDuckGo, Tavily, etc.)
def mock_search_api(query: str, num_results: int) -> Dict[str, Any]:
    """
    Mocks a call to an external search API.
    """
    logger.info(f"Mock search API called with query: '{query}', num_results: {num_results}")
    results = []
    for i in range(num_results):
        results.append({
            "title": f"Mock Search Result {i+1} for '{query}'",
            "link": f"https://example.com/search?q={query.replace(' ', '+')}&page={i+1}",
            "snippet": f"This is a mock snippet for search result {i+1} related to '{query}'. "
                       "It contains some descriptive text about the content found at the link."
        })
    return {"query": query, "results": results}


class SearchTool(BaseTool):
    """
    A tool to perform a web search using a (mocked) search engine.
    """
    name: str = "web_search"
    description: str = (
        "Performs a web search for the given query and returns a list of results. "
        "Useful for finding information on the internet, current events, or specific topics."
    )
    args_schema: Type[BaseModel] = SearchToolInput

    # You could add configuration here, e.g., API keys for a real search engine
    # def __init__(self, api_key: Optional[str] = None, **kwargs):
    #     super().__init__(**kwargs)
    #     self.api_key = api_key
    #     if not self.api_key:
    #         logger.warning("SearchTool initialized without an API key. Mock search will be used.")


    def _run(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Executes the search.

        Args:
            query (str): The search query.
            num_results (int): The number of results to return.

        Returns:
            Dict[str, Any]: A dictionary containing the search query and a list of results,
                            where each result has a title, link, and snippet.
                            Returns an error message string if the search fails.
        """
        logger.info(f"Executing SearchTool with query: '{query}', num_results: {num_results}")
        try:
            # In a real tool, you would use self.api_key and make a request
            # For example, using the 'requests' library:
            # response = requests.get(
            #     "https://api.somesearch_engine.com/v1/search",
            #     params={"q": query, "num": num_results},
            #     headers={"Authorization": f"Bearer {self.api_key}"}
            # )
            # response.raise_for_status() # Raise an exception for HTTP errors
            # search_results = response.json()

            # Using the mock function for this example
            search_results = mock_search_api(query, num_results)

            if not search_results or "results" not in search_results or not search_results["results"]:
                return {"query": query, "results": [], "message": "No results found."}

            return search_results

        except requests.exceptions.RequestException as e:
            logger.error(f"SearchTool: HTTP Request error for query '{query}': {e}", exc_info=True)
            return {"error": f"Network error during search: {e}"}
        except Exception as e:
            logger.error(f"SearchTool: Unexpected error for query '{query}': {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during search: {e}"}

    # Example of how an async version might look (if the API supports it)
    async def _arun(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        logger.info(f"Async executing SearchTool with query: '{query}', num_results: {num_results}")
        # If using httpx for async requests:
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
        #             return {"query": query, "results": [], "message": "No results found."}
        #         return search_results
        #     except httpx.RequestError as e:
        #         logger.error(f"SearchTool (async): HTTP Request error for query '{query}': {e}", exc_info=True)
        #         return {"error": f"Network error during async search: {e}"}
        #     except Exception as e:
        #         logger.error(f"SearchTool (async): Unexpected error for query '{query}': {e}", exc_info=True)
        #         return {"error": f"An unexpected error occurred during async search: {e}"}

        # For this example, we'll just call the mock sync version (not truly async)
        # To make it truly async, mock_search_api would also need to be async.
        # import asyncio
        # await asyncio.sleep(0.05) # Simulate async delay
        return self._run(query, num_results) # Fallback to sync for mock


if __name__ == '__main__':
    from configs.config import load_config
    from configs.logging_config import setup_logging
    import asyncio

    load_config()
    setup_logging()

    logger.info("Testing SearchTool...")
    search_tool = SearchTool() # Add api_key="YOUR_KEY" if using a real one

    # Test with valid input
    results1 = search_tool.run(query="latest AI advancements", num_results=2)
    logger.info(f"SearchTool results (latest AI advancements, 2): {results1}")
    assert "results" in results1
    assert len(results1["results"]) == 2
    assert "Mock Search Result 1" in results1["results"][0]["title"]

    # Test with default num_results
    results2 = search_tool.run(query="python programming tips")
    logger.info(f"SearchTool results (python programming tips, default 3): {results2}")
    assert "results" in results2
    assert len(results2["results"]) == 3

    # Test using Pydantic model input
    input_data = SearchToolInput(query="best pizza recipe", num_results=1)
    results3 = search_tool.run(input_data)
    logger.info(f"SearchTool results (Pydantic input): {results3}")
    assert "results" in results3
    assert len(results3["results"]) == 1
    assert "pizza recipe" in results3["results"][0]["title"]

    # Test invalid input (e.g., num_results out of bounds, handled by Pydantic)
    invalid_results = search_tool.run(query="test", num_results=20) # ge=1, le=10
    logger.info(f"SearchTool results (invalid num_results): {invalid_results}")
    assert "Error: Input validation failed" in str(invalid_results)

    # Test async run
    async def run_async_search():
        async_results = await search_tool.arun(query="async search test", num_results=1)
        logger.info(f"SearchTool async results: {async_results}")
        assert "results" in async_results
        assert len(async_results["results"]) == 1
        assert "async search test" in async_results["results"][0]["title"]

    asyncio.run(run_async_search())

    logger.info(f"SearchTool Schema: {SearchTool.get_tool_info()}")
    logger.info("SearchTool tests completed successfully.")

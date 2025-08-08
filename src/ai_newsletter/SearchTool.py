from langchain.tools import BaseTool
from ai_newsletter.SearchTool import TavilyClient
from pydantic import PrivateAttr
import logging
import json
from typing import TypedDict, Optional, Dict, Any

class TavilySearchTool(BaseTool):
    name: str = "Tavily Web Search"
    description: str = "Searches the web for the latest AI/ML news and trends using the Tavily API."
    _client: Any = PrivateAttr()
    _search_depth: str = PrivateAttr()
    _max_result: int = PrivateAttr()
    _include_answer: bool = PrivateAttr()
    _include_images: bool = PrivateAttr()
    _timeout: int = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        search_depth: str = "advanced",
        max_result: int = 10,
        include_answer: bool = True,
        include_images: bool = False,
        timeout: int = 60,
    ):
        super().__init__()
        self._client = TavilyClient(api_key=api_key)
        self._search_depth = search_depth
        self._max_result = max_result
        self._include_answer = include_answer
        self._include_images = include_images
        self._timeout = timeout

    def _run(self, query: str) -> str:
        try:
            response = self._client.search(
                query=query,
                search_depth=self._search_depth,
                num_results=self._max_result,
                include_answer=self._include_answer,
                include_images=self._include_images,
                timeout=self._timeout,
            )
            results = response.get("results", [])
            if not results:
                return "No results found."
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error during Tavily search: {e}")
            return f"Error during search: {str(e)}"

    def _call(self, query: str) -> str:
        return self._run(query)
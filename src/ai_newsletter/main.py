import json
from langchain.tools import BaseTool
from tavily import TavilyClient
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import PrivateAttr

from typing import TypedDict, Optional

class NewsletterState(TypedDict):
    query: str
    research_summary: Optional[str]
    analysis: Optional[str]
    newsletter: Optional[str]
    final_newsletter: Optional[str]

# --- Custom Tavily Search Tool using TavilyClient ---
class TavilySearchTool(BaseTool):
    name: str = "Tavily Web Search"
    description: str = "Searches the web for the latest AI/ML news and trends using the Tavily API."
    _client: any = PrivateAttr()
    _search_depth: any = PrivateAttr()
    _max_result: any = PrivateAttr()
    _include_answer: any = PrivateAttr()
    _include_images: any = PrivateAttr()
    _timeout: any = PrivateAttr()

    def __init__(self, api_key, search_depth="advanced", max_result=10, include_answer=True, include_images=False, timeout=60):
        super().__init__()
        self._client = TavilyClient(api_key=api_key)
        self._search_depth = search_depth
        self._max_result = max_result
        self._include_answer = include_answer
        self._include_images = include_images
        self._timeout = timeout

    def _run(self, query: str):
        try:
            response = self._client.search(
                query=query,
                search_depth=self._search_depth,
                num_results=self._max_result,
                include_answer=self._include_answer,
                include_images=self._include_images,
                timeout=self._timeout
            )
            results = response.get("results", [])
            if not results:
                return "No results found."
            # Return as JSON string for LLM parsing
            import json
            return json.dumps(results[:5], ensure_ascii=False)
        except Exception as e:
            return f"Error during search: {str(e)}"

    def _call(self, query: str):
        # This method is required by BaseTool to run synchronously
        return self._run(query)


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:latest",
    temperature=0.3,
)


def research_agent(input_dict):
    # input_dict: {"query": ..., "tavily_tool": TavilySearchTool}
    query = input_dict["query"]
    tavily_tool = input_dict["tavily_tool"]
    tool_output = tavily_tool.run(query)
    # LLM prompt
    prompt = (
        "You will receive a JSON list of news items. For each, extract the title, url, and summary. "
        "If the list is empty, say 'No news found.'\n"
        f"JSON: {tool_output}\n"
        "Summarize the 5 most important items in markdown bullet points."
    )
    return llm.invoke(prompt)

def analysis_agent(input_dict):
    research_summary = input_dict["research_summary"]
    prompt = (
        "Analyze the following AI/ML news research summary. Identify the most significant developments and trends, "
        "explain their implications for different industries, and provide a concise executive summary:\n"
        f"{research_summary}"
    )
    return llm.invoke(prompt)

def newsletter_writer_agent(input_dict):
    analysis = input_dict["analysis"]
    prompt = (
        "Write a professional, engaging AI/ML newsletter based on the following analysis. "
        "Include a catchy intro, organize the main stories with summaries and links, highlight trends, and end with a closing remark. "
        "Format in markdown for readability:\n"
        f"{analysis}"
    )
    return llm.invoke(prompt)

def editor_agent(input_dict):
    newsletter = input_dict["newsletter"]
    prompt = (
        "Edit the following AI/ML newsletter draft for clarity, accuracy, professionalism, and markdown formatting. "
        "Ensure it is ready for publication:\n"
        f"{newsletter}"
    )
    return llm.invoke(prompt)


# Define the state keys for each step
def build_graph(tavily_tool):
    workflow = StateGraph(NewsletterState)
    # Step 1: Research
    workflow.add_node(
        "research",
        lambda state: {
            "research_summary": research_agent({"query": state["query"], "tavily_tool": tavily_tool})
        }
    )
    # Step 2: Analysis
    workflow.add_node(
        "analysis",
        lambda state: {
            "analysis": analysis_agent({"research_summary": state["research_summary"]})
        }
    )
    # Step 3: Newsletter Writing
    workflow.add_node(
        "newsletter",
        lambda state: {
            "newsletter": newsletter_writer_agent({"analysis": state["analysis"]})
        }
    )
    # Step 4: Editing
    workflow.add_node(
        "edit",
        lambda state: {
            "final_newsletter": editor_agent({"newsletter": state["newsletter"]})
        }
    )

    # Edges: research -> analysis -> newsletter -> edit -> END
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "newsletter")
    workflow.add_edge("newsletter", "edit")
    workflow.add_edge("edit", END)

    # Set entry point
    workflow.set_entry_point("research")
    return workflow.compile()

def main():
    import os
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    tavily_tool = TavilySearchTool(api_key=TAVILY_API_KEY)

    graph = build_graph(tavily_tool)
    # Start the workflow with a query
    query = (
        "latest AI and machine learning news 2025, machine learning breakthroughs, AI funding news, "
        "AI regulation updates, new AI models released, AI industry trends, OpenAI news, Google AI updates, "
        "Microsoft AI updates, NVIDIA AI updates"
    )
    result = graph.invoke({"query": query})
    print("\n\n===== FINAL NEWSLETTER =====\n")
    print(result["final_newsletter"])

if __name__ == "__main__":
    main()
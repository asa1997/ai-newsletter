from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
from SearchTool import TavilySearchTool
from langchain_community.chat_models import ChatOllama

class NewsletterState(TypedDict):
    query: str
    research_summary: Optional[str]
    analysis: Optional[str]
    newsletter: Optional[str]
    final_newsletter: Optional[str]
    email_status: Optional[str]

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:latest",
    temperature=0.3,
)
RESEARCH_PROMPT = (
    "You will receive a JSON list of news items. For each, extract the title, url, summary, and source. "
    "If the list is empty, say 'No news found.'\n"
    "JSON: {tool_output}\n"
    "Summarize the 5 most important items in markdown bullet points, including citations and source URLs."
)

ANALYSIS_PROMPT = (
    "Analyze the following {topic} security news research summary. Identify the most significant developments and trends, "
    "explain their implications for {topic} security and related fields, provide a concise executive summary, and categorize "
    "them into the following categories: LLM Security, Agentic Threats, Vulnerability Disclosure, Security Tools, Academic Research.\n"
    "{research_summary}"
)

NEWSLETTER_PROMPT = (
    "Write a professional, engaging {topic} security newsletter based on the following analysis. "
    "Include a catchy intro, organize the main stories with summaries, categories, citations, and links, highlight trends, and end with a closing remark. "
    "Format in markdown for readability:\n"
    "{analysis}"
)

EDITOR_PROMPT = (
    "Edit the following {topic} security newsletter draft for clarity, accuracy, professionalism, and markdown formatting. "
    "Ensure it is ready for publication:\n"
    "{newsletter}"
)


def research_agent(input_dict: Dict[str, Any]) -> str:
    """
    Perform research by running a web search and summarizing results.

    Args:
        input_dict: Dictionary with keys 'query' (str) and 'tavily_tool' (TavilySearchTool).

    Returns:
        A markdown summary of the top news items.
    """
    query = input_dict["query"]
    tavily_tool = input_dict["tavily_tool"]
    tool_output = tavily_tool.run(query)
    prompt = RESEARCH_PROMPT.format(tool_output=tool_output)
    return llm.invoke(prompt)


def analysis_agent(input_dict: Dict[str, Any]) -> str:
    """
    Analyze the research summary to identify key trends and implications.

    Args:
        input_dict: Dictionary with key 'research_summary' (str).

    Returns:
        An executive summary of the analysis.
    """
    research_summary = input_dict["research_summary"]
    prompt = ANALYSIS_PROMPT.format(research_summary=research_summary)
    return llm.invoke(prompt)


def newsletter_writer_agent(input_dict: Dict[str, Any]) -> str:
    """
    Write a newsletter based on the analysis.

    Args:
        input_dict: Dictionary with key 'analysis' (str).

    Returns:
        A markdown formatted newsletter or JSON structured output for dashboard integration.
    """
    analysis = input_dict["analysis"]
    prompt = NEWSLETTER_PROMPT.format(analysis=analysis)
    return llm.invoke(prompt)


def editor_agent(input_dict: Dict[str, Any]) -> str:
    """
    Edit the newsletter draft for clarity and professionalism.

    Args:
        input_dict: Dictionary with key 'newsletter' (str).

    Returns:
        The final edited newsletter.
    """
    newsletter = input_dict["newsletter"]
    prompt = EDITOR_PROMPT.format(newsletter=newsletter)
    return llm.invoke(prompt)

def build_graph(tavily_tool: TavilySearchTool) -> StateGraph:
    """
    Build the newsletter generation workflow graph.

    Args:
        tavily_tool: An instance of TavilySearchTool.

    Returns:
        A compiled StateGraph representing the workflow.
    """
    workflow = StateGraph(NewsletterState)

    workflow.add_node(
        "research_agent",
        lambda state: {
            "research_summary": research_agent(
                {"query": state["query"], "tavily_tool": tavily_tool}
            )
        },
    )
    workflow.add_node(
        "analysis_agent",
        lambda state: {
            "analysis": analysis_agent({"research_summary": state["research_summary"]})
        },
    )
    workflow.add_node(
        "newsletter_agent",
        lambda state: {
            "newsletter": newsletter_writer_agent({"analysis": state["analysis"]})
        },
    )
    workflow.add_node(
        "editor_agent",
        lambda state: {"final_newsletter": editor_agent({"newsletter": state["newsletter"]})},
    )
    # workflow.add_node(
    #     "send_email",
    #     lambda state: {"email_status": email_sender_agent({"final_newsletter": state["final_newsletter"]})},
    # )

    workflow.add_edge("research_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "newsletter_agent")
    workflow.add_edge("newsletter_agent", "editor_agent")
    # workflow.add_edge("editor_agent", "send_email")
    workflow.add_edge("editor_agent", END)

    workflow.set_entry_point("research_agent")
    return workflow.compile()


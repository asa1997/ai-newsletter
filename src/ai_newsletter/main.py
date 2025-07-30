import json
import os
import argparse
import logging
from typing import TypedDict, Optional, Dict, Any

from langchain.tools import BaseTool
from tavily import TavilyClient
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import PrivateAttr
import smtplib
from email.message import EmailMessage


class NewsletterState(TypedDict):
    query: str
    research_summary: Optional[str]
    analysis: Optional[str]
    newsletter: Optional[str]
    final_newsletter: Optional[str]
    email_status: Optional[str]


# --- Custom Tavily Search Tool using TavilyClient ---
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


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:latest",
    temperature=0.3,
)


# Prompt templates as constants for easier maintenance
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


def email_sender_agent(input_dict: Dict[str, Any]) -> str:
    """
    Send the final newsletter as an email.

    Args:
        input_dict: Dictionary with key 'final_newsletter' (str).

    Returns:
        Status message indicating success or failure.
    """
    newsletter_content = input_dict.get("final_newsletter", "")
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_username = os.environ.get("SMTP_USERNAME")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    email_sender = os.environ.get("EMAIL_SENDER")
    email_recipient = os.environ.get("EMAIL_RECIPIENT")

    if not all([smtp_server, smtp_port, smtp_username, smtp_password, email_sender, email_recipient]):
        error_msg = "SMTP configuration is incomplete. Please set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_SENDER, and EMAIL_RECIPIENT environment variables."
        logging.error(error_msg)
        return error_msg

    msg = EmailMessage()
    msg["Subject"] = "Weekly AI/ML Newsletter"
    msg["From"] = email_sender
    msg["To"] = email_recipient
    msg.set_content(newsletter_content)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        success_msg = f"Email sent successfully to {email_recipient}."
        logging.info(success_msg)
        return success_msg
    except Exception as e:
        error_msg = f"Failed to send email: {e}"
        logging.error(error_msg)
        return error_msg


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
    workflow.add_node(
        "send_email",
        lambda state: {"email_status": email_sender_agent({"final_newsletter": state["final_newsletter"]})},
    )

    workflow.add_edge("research_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "newsletter_agent")
    workflow.add_edge("newsletter_agent", "editor_agent")
    # workflow.add_edge("editor_agent", "send_email")
    workflow.add_edge("editor_agent", END)

    workflow.set_entry_point("research_agent")
    return workflow.compile()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="AI Newsletter Generator")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query string for the newsletter research step.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        choices=["AI", "PQC"],
        default="AI",
        help="Topic to query for: AI or PQC (Post-Quantum Cryptography). Default is AI.",
    )
    parser.add_argument(
        "--search-depth",
        type=str,
        default=os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
        help="Search depth for TavilySearchTool (default: advanced).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=int(os.getenv("TAVILY_MAX_RESULTS", "10")),
        help="Maximum number of search results (default: 10).",
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        help="Include answers in search results (default: True).",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include images in search results (default: False).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("TAVILY_TIMEOUT", "60")),
        help="Timeout for search requests in seconds (default: 60).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the AI newsletter generator.
    """
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        logging.error("TAVILY_API_KEY environment variable is not set.")
        return

    tavily_tool = TavilySearchTool(
        api_key=TAVILY_API_KEY,
        search_depth=args.search_depth,
        max_result=args.max_results,
        include_answer=args.include_answer,
        include_images=args.include_images,
        timeout=args.timeout,
    )

    if args.query:
        query = args.query
    else:
        if args.topic == "AI":
            query = (
                "Latest AI security news, vulnerabilities, benchmarks, and tools related to AI security, security for AI, and post-quantum cryptography (PQC). "
                "Include topics on LLM Security, Agentic Threats, Vulnerability Disclosure, Security Tools, and Academic Research. "
                "Focus on sources like NIST, MLCommons, arXiv, Hacker News, security mailing lists, and relevant industry reports."
            )
        else:  # PQC
            query = (
                "Latest post-quantum cryptography (PQC) news, research, vulnerabilities, benchmarks, and tools. "
                "Include topics on PQC algorithms, standardization efforts, security analysis, implementation challenges, and academic research. "
                "Focus on sources like NIST PQC Project, IBM Research Blog, Cloudflare Blog (Crypto), Google Security Blog, arXiv (cryptography)."
            )

    logging.info("Starting AI newsletter generation workflow.")
    logging.info(f"Using query: {query}")
    logging.info(f"Topic selected: {args.topic}")

    # Format prompts with the topic
    global RESEARCH_PROMPT, ANALYSIS_PROMPT, NEWSLETTER_PROMPT, EDITOR_PROMPT
    RESEARCH_PROMPT = (
        "You will receive a JSON list of news items. For each, extract the title, url, summary, and source. "
        "If the list is empty, say 'No news found.'\n"
        "JSON: {tool_output}\n"
        "Summarize the 5 most important items in markdown bullet points, including citations and source URLs."
    )
    ANALYSIS_PROMPT = (
        f"Analyze the following {args.topic} security news research summary. Identify the most significant developments and trends, "
        f"explain their implications for {args.topic} security and related fields, provide a concise executive summary, and categorize "
        "them into the following categories: LLM Security, Agentic Threats, Vulnerability Disclosure, Security Tools, Academic Research.\n"
        "{research_summary}"
    )
    NEWSLETTER_PROMPT = (
        f"Write a professional, engaging {args.topic} security newsletter based on the following analysis. "
        "Include a catchy intro, organize the main stories with summaries, categories, citations, and links, highlight trends, and end with a closing remark. "
        "Format in markdown for readability:\n"
        "{analysis}"
    )
    EDITOR_PROMPT = (
        f"Edit the following {args.topic} security newsletter draft for clarity, accuracy, professionalism, and markdown formatting. "
        "Ensure it is ready for publication:\n"
        "{newsletter}"
    )

    try:
        graph = build_graph(tavily_tool)
        result = graph.invoke({"query": query})
        newsletter = result.get("final_newsletter", "")

        logging.info("\n\n===== FINAL NEWSLETTER =====\n")
        if hasattr(newsletter, "content"):
            print(newsletter.content)
        else:
            print(newsletter)
    except Exception as e:
        logging.error(f"Error during newsletter generation: {e}")



# def main() -> None:
#     """
#     Main entry point for the AI newsletter generator.
#     """
#     args = parse_args()

#     logging.basicConfig(
#         level=logging.DEBUG if args.verbose else logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#     )

#     TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
#     if not TAVILY_API_KEY:
#         logging.error("TAVILY_API_KEY environment variable is not set.")
#         return

#     tavily_tool = TavilySearchTool(
#         api_key=TAVILY_API_KEY,
#         search_depth=args.search_depth,
#         max_result=args.max_results,
#         include_answer=args.include_answer,
#         include_images=args.include_images,
#         timeout=args.timeout,
#     )

#     # Updated default query to focus on AI security news, vulnerabilities, benchmarks, tools, PQC, and related topics
#     default_query = (
#         "Latest AI security news, vulnerabilities, benchmarks, and tools related to AI security, security for AI, and post-quantum cryptography (PQC). "
#         "Include topics on LLM Security, Agentic Threats, Vulnerability Disclosure, Security Tools, and Academic Research. "
#         "Focus on sources like NIST, MLCommons, arXiv, Hacker News, security mailing lists, OWASP, MITRE (ATLAS/ATT&CK), AI Snake Oil, Alignment Newsletter, Security & Safety Substacks, Ben Dickson, HuggingFace SafetAI  and relevant industry reports."
#     )

#     query = args.query if args.query else default_query

#     logging.info("Starting AI newsletter generation workflow.")
#     logging.info(f"Using query: {query}")

#     try:
#         graph = build_graph(tavily_tool)
#         result = graph.invoke({"query": query})
#         newsletter = result.get("final_newsletter", "")

#         logging.info("\n\n===== FINAL NEWSLETTER =====\n")
#         if hasattr(newsletter, "content"):
#             print(newsletter.content)
#         else:
#             print(newsletter)
#     except Exception as e:
#         logging.error(f"Error during newsletter generation: {e}")


if __name__ == "__main__":
    main()

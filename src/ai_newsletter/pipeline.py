import logging
import os
from SearchTool import TavilySearchTool
from MakeGraph import build_graph

def trigger(query: str = "", topic: str = "AI") -> None:
    """
    Main entry point for the AI newsletter generator.
    """
    # args = parse_args()

    # logging.basicConfig(
    #     level=logging.DEBUG if args.verbose else logging.INFO,
    #     format="%(asctime)s - %(levelname)s - %(message)s",
    # )

    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        logging.error("TAVILY_API_KEY environment variable is not set.")
        return

    tavily_tool = TavilySearchTool(
        api_key=TAVILY_API_KEY,
        # search_depth=args.search_depth,
        search_depth="advanced",
        # max_result=args.max_results,
        max_result=10,
        # include_answer=args.include_answer,
        include_answer=True,
        # include_images=args.include_images,
        include_images=False,
        # timeout=args.timeout,
        timeout=60,

    )

    # if args.query:
    if query:
        query = query
    else:
        if topic == "AI":
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
    logging.info(f"Topic selected: {topic}")

    # # Format prompts with the topic
    # global RESEARCH_PROMPT, ANALYSIS_PROMPT, NEWSLETTER_PROMPT, EDITOR_PROMPT
    # RESEARCH_PROMPT = (
    #     "You will receive a JSON list of news items. For each, extract the title, url, summary, and source. "
    #     "If the list is empty, say 'No news found.'\n"
    #     "JSON: {tool_output}\n"
    #     "Summarize the 5 most important items in markdown bullet points, including citations and source URLs."
    # )
    # ANALYSIS_PROMPT = (
    #     f"Analyze the following {topic} security news research summary. Identify the most significant developments and trends, "
    #     f"explain their implications for {topic} security and related fields, provide a concise executive summary, and categorize "
    #     "them into the following categories: LLM Security, Agentic Threats, Vulnerability Disclosure, Security Tools, Academic Research.\n"
    #     "{research_summary}"
    # )
    # NEWSLETTER_PROMPT = (
    #     f"Write a professional, engaging {topic} security newsletter based on the following analysis. "
    #     "Include a catchy intro, organize the main stories with summaries, categories, citations, and links, highlight trends, and end with a closing remark. "
    #     "Format in markdown for readability:\n"
    #     "{analysis}"
    # )
    # EDITOR_PROMPT = (
    #     f"Edit the following {topic} security newsletter draft for clarity, accuracy, professionalism, and markdown formatting. "
    #     "Ensure it is ready for publication:\n"
    #     "{newsletter}"
    # )

    try:
        graph = build_graph(tavily_tool)
        result = graph.invoke({"query": query})
        newsletter = result.get("final_newsletter", "")

        logging.info("\n\n===== FINAL NEWSLETTER =====\n")
        if hasattr(newsletter, "content"):
            # print(newsletter.content)
            return newsletter.content
        else:
            # print(newsletter)
            return newsletter
    except Exception as e:
        logging.error(f"Error during newsletter generation: {e}")


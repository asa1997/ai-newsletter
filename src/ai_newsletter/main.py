import os
import traceback
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import BaseTool
from tavily import TavilyClient
from pydantic import PrivateAttr

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

    def _run(self, query: str) -> str:
        try:
            results = self.client.search(
                query=query,
                search_depth=self.search_depth,
                num_results=self.max_result,
                include_answer=self.include_answer,
                include_images=self.include_images,
                timeout=self.timeout
            )
            # Format results for LLM consumption
            formatted = []
            for item in results:
                title = item.get("title") or ""
                url = item.get("url") or ""
                content = item.get("content") or ""
                formatted.append(f"- {title}\n  {url}\n  {content}")
            return "\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            return f"Tavily search failed: {str(e)}"

# --- Main CrewAI Workflow ---
def main():
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")

    tavilySearch = TavilySearchTool(
        api_key=TAVILY_API_KEY,
        search_depth="advanced",
        max_result=10,
        include_answer=True,
        include_images=False,
        timeout=60
    )

    ollamaLLM = LLM(
        model="ollama/llama3.2:latest",
        base_url="http://localhost:11434",
        temperature=0.3
    )

    researchAgent = Agent(
        role="AI/ML Research Specialist",
        goal="Find and gather the latest, most significant news and trends in AI and Machine Learning using web search",
        backstory="""You are a senior research analyst specializing in artificial intelligence and machine learning. You have a keen eye for identifying breakthrough technologies, emerging trends and significant industry developments. You excel at finding credible sources and distinguishing between hype and genuine innovation. You use advanced web search tools to gather comprehensive information from multiple sources.""",
        tools=[tavilySearch],
        llm=ollamaLLM,
        verbose=True,
        allow_delegation=False
    )

    analysisAgent = Agent(
        role="AI/ML Trend Analyst",
        goal="Analyze research findings to identify key patterns, implications and future trends",
        backstory="""You are an expert analyst with deep knowledge of AI and ML technologies. You excel at connecting dots between different developments, understanding their business implications, predicting their future trends. You have a talent for synthesizing complex technical information into clear insights""",
        llm=ollamaLLM,
        verbose=True,
        allow_delegation=False
    )

    newsLetterWriter = Agent(
        role="Professional Newsletter Writer",
        goal="Create engaging, professional newsletters that inform and educate readers about AI/ML developments",
        backstory="""You are a skilled technical writer with expertise in creating compelling newsletters for technology professionals. You know how to structure the content for maximum readability, engagement and value. You excel at translating complex technical concepts into accessible language while maintaining accuracy.""",
        llm=ollamaLLM,
        verbose=True,
        allow_delegation=False,
    )

    editorAgent = Agent(
        role="Newsletter Editor",
        goal="Ensure newsletter quality, accuracy and professional presentation",
        backstory="""You are an experienced editor with a background in technology publishing. You have a keen eye for detail, excellent grammar skills and deep understanding of what makes content engaging and professional. You ensure consistency, accuracy and high editorial standards.""",
        llm=ollamaLLM,
        verbose=True,
        allow_delegation=False
    )

    researchTask = Task(
        description="""Conduct comprehensive research on latest AI and ML news and trends using web search.
        Focus on:
        1. Recent breakthrough technologies and research papers (last 7-14 days)
        2. Major industry announcements and product launches
        3. Funding rounds and acquisitions in AI/ML space
        4. Regulatory developments affecting AI/ML
        5. Emerging applications and use cases
        6. Key conferences, events and expert opinions
        7. Notable AI model releases or updates
        8. Industry partnerships and collaborations

        Search Strategy:
        - Use multiple search queries to cover different aspects
        - Search for "latest AI news 2025", "machine learning breakthroughs", "AI funding news"
        - Look for "AI regulation updates", "new AI models released", "AI industry trends"
        - Search for specific companies like "OpenAI news", "Google AI updates", "Microsoft AI updates", "NVIDIA AI updates"
        - Include academic sources and research publications

        Prioritize credible sources like major tech publications, research institutions and official company announcements.""",
        expected_output="""A comprehensive research report containing:
        - 15-20 significant AI/ML news items with sources and dates
        - Key trends and patterns identified across the findings
        - Important quotes, statistics and data points
        - Categorized information by topic (research, industry, regulation, applications, funding)
        - Sources credibility assessment for each finding
        - Summary of the most impactful developments""",
        agent=researchAgent
    )

    analysisTask = Task(
        description="""Analyze the research findings to identify the most significant developments and trends. Your analysis should:
        1. Identify top 7-10 most important stories based on impact and relevance
        2. Analyze the implications of these developments for different stakeholders
        3. Connect related stories to show broader trends and patterns
        4. Assess the potential impact on various industries (healthcare, finance, education)
        5. Identify emerging patterns and future predictions
        6. Highlight any controversial or debated topics
        7. Evaluate the credibility and significance of each development
        8. Provide context for why each story matters

        Consider:
        - Short-term vs long-term implications
        - Technical vs business impact
        - Regional vs global significance
        - Competitive landscape changes""",
        expected_output="""An analytical report containing:
        - Top 7-10 priority stories with detailed impact analysis
        - Comprehensive trend analysis and implications
        - Industry impact assessment across multiple sectors
        - Future outlook and predictions with reasoning
        - Risk and opportunity assessment
        - Recommended focus areas for newsletter
        - Executive summary for key insights""",
        agent=analysisAgent,
        context=[researchTask]
    )

    newsletterTask = Task(
        description="""Write a professional, engaging AI/ML newsletter based on the analytical report and research findings. The newsletter should:
        - Start with a catchy introduction summarizing the week's AI/ML landscape
        - Present the top stories with clear headlines, concise summaries, and source links
        - Include insightful commentary and context for each story
        - Highlight key trends, statistics, and quotes
        - Organize content logically (e.g., by topic or impact)
        - Conclude with a forward-looking closing remark or call to action
        - Use markdown formatting for readability
        - Target an audience of technology professionals and enthusiasts""",
        expected_output="""A markdown-formatted newsletter ready for publication, including:
        - Title and introduction
        - 7-10 main stories with summaries and links
        - Trend highlights and expert commentary
        - Closing section
        - Professional, engaging, and accessible language""",
        agent=newsLetterWriter,
        context=[analysisTask]
    )

    editorTask = Task(
        description="""Review and edit the drafted AI/ML newsletter for:
        - Clarity, accuracy, and professionalism
        - Consistent tone and style
        - Correct grammar, spelling, and formatting
        - Factual accuracy and credible sourcing
        - Engaging and accessible presentation

        Make any necessary corrections or improvements to ensure the newsletter meets high editorial standards and is ready for distribution.""",
        expected_output="""A final, polished markdown newsletter that is publication-ready and meets all editorial standards.""",
        agent=editorAgent,
        context=[newsletterTask]
    )

    crew = Crew(
        agents=[researchAgent, analysisAgent, newsLetterWriter, editorAgent],
        tasks=[researchTask, analysisTask, newsletterTask, editorTask],
        process=Process.sequential,
        verbose=True,
        memory=False,
        embedder={
            "provider": "ollama",
            "config": {
                "model": "mxbai-embed-large:latest",
                "url": "http://localhost:11434/api/embeddings"
            }
        },
        planning=False
    )
    try:
        result = crew.kickoff()
        print(result)
    except Exception as e:
        print("An error occurred while running the crew:")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
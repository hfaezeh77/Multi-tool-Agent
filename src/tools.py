import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from rag_chain import get_retriever

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Reuse the Project 1 retriever, wrapped as a tool
_retriever = get_retriever()

@tool
def search_my_documents(query: str) -> str:
    """Search the user's personal LangChain/LangGraph documents for information.
    Use this for questions about LangChain, LangGraph, StateGraph, agents,
    or anything likely covered in the indexed documentation.
    Do NOT use this for current events, prices, or general web knowledge."""
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant information found in the documents."
    return "\n\n".join(d.page_content for d in docs)


# Web search tool
web_search = TavilySearch(max_results=3)
web_search.name = "search_the_web"
web_search.description = (
    "Search the live web for current events, recent news, prices, or any "
    "factual information NOT likely to be in the user's personal documents. "
    "Do NOT use this for questions about LangChain/LangGraph concepts — "
    "use search_my_documents for those instead."
)

TOOLS = [search_my_documents, web_search]
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from tools import TOOLS

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
).bind_tools(TOOLS)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools. "
    "If a tool is relevant, call it. If not, answer directly and clearly in plain text. "
    "Never return an empty response."
))

def agent_node(state: State):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)

    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  [tool call] {tc['name']}({tc['args']})")

    return {"messages": [response]}


def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def build_graph():
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after a tool runs, go back to the agent to decide next step

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "agent-session-1"}}

    while True:
        q = input("\nYou: ")
        if q.lower() == "quit":
            break
        result = app.invoke({"messages": [HumanMessage(content=q)]}, config=config)
        answer = result["messages"][-1].content

        if not answer.strip():
            print("AI: (received an empty response — try rephrasing, or ask again)")
        else:
            print("AI:", answer)
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from rag_chain import build_rag_chain

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

rag_chain = build_rag_chain()

def rag_node(state: State):
    messages = state["messages"]

    # Build a simple history string from all messages so far
    history_text = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
        for m in messages[:-1]  # all messages except the current one
    )
    current_question = messages[-1].content

    # Combine history + current question into one input for the RAG chain
    full_input = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current question: {current_question}"
        if history_text else current_question
    )

    answer = rag_chain.invoke(full_input)
    return {"messages": [("ai", answer)]}

def build_graph():
    graph = StateGraph(State)
    graph.add_node("rag", rag_node)
    graph.set_entry_point("rag")
    graph.add_edge("rag", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        q = input("\nYou: ")
        if q.lower() == "quit":
            break
        result = app.invoke({"messages": [HumanMessage(content=q)]}, config=config)
        print("AI:", result["messages"][-1].content)
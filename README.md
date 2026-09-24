# Multi-Tool Agent (LangChain + LangGraph)

A LangGraph agent that decides, per question, whether to search a personal document set, search the live web, or answer directly — instead of following a fixed pipeline. Built as Project 2 of a 3-project series while working through *Building AI and LLM Applications with LangChain and LangGraph* (Oshin & Campos), extending Project 1 (RAG Chatbot with Memory).

## What it does

- Exposes two tools to an LLM: a document retriever (reused from Project 1) and a live web search tool
- Lets the LLM decide, based on each question, whether to call a tool, which one, or skip tools entirely and answer directly
- Loops: after a tool runs, control returns to the agent so it can call another tool or produce a final answer
- Retains conversational memory across turns via the same checkpointer pattern as Project 1
- Logs which tool(s) were used for each response, for transparency into the agent's decisions

## Architecture

```
User input
   │
   ▼
StateGraph (LangGraph)
   │  state: { messages: [...] }  ← persisted via checkpointer, keyed by thread_id
   ▼
agent node (LLM bound to tools)
   │
   ├── decides: call a tool? ──► should_continue() checks response.tool_calls
   │
   ├── YES → tools node (ToolNode) ──► executes the chosen tool ──► back to agent node
   │
   └── NO  → END (final answer returned)
```

This is a loop, not a fixed pipeline (the key structural difference from Project 1): the agent can call zero, one, or multiple tools in sequence before deciding it has enough information to answer.

## Tools

| Tool | Purpose | Source |
|---|---|---|
| `search_my_documents` | Searches the indexed LangChain/LangGraph documentation (from Project 1) | Local Chroma vector store |
| `search_the_web` | Searches the live web for current events, prices, or anything not in the local docs | Tavily API |

Each tool's docstring/description is written specifically to help the LLM distinguish when to use it over the other — tool descriptions function as the *entire* basis for the agent's routing decision, since the LLM never sees the underlying code.

## Tech stack

Same as Project 1, plus:

| Component | Tool |
|---|---|
| Web search | Tavily (`langchain-tavily`), free tier |
| Tool orchestration | LangGraph `ToolNode`, conditional edges |

## Project structure

```
langgraph-projects/
├── data/
├── src/
│   ├── indexing.py
│   ├── rag_chain.py
│   ├── memory_graph.py      # Project 1
│   ├── tools.py              # tool definitions (NEW)
│   └── agent_graph.py        # Project 2 entry point (NEW)
├── .env
├── .env.example
└── requirements.txt
```

## Setup

Builds on the Project 1 environment — if you already have that set up, you only need:

```bash
pip install langchain-tavily
pip freeze > requirements.txt
```

Add a Tavily key to `.env` (free tier at https://tavily.com):
```
OPENROUTER_API_KEY=sk-or-v1-...
TAVILY_API_KEY=tvly-...
```

## Usage

```bash
python src/agent_graph.py
```

Type `quit` to exit.

## Example interaction

```
You: hi
AI: Hi! How can I help?

You: What is a StateGraph in LangGraph?
  [tool call] search_my_documents({'query': 'StateGraph in LangGraph'})
  [tool call] search_my_documents({'query': 'StateGraph definition and purpose in LangGraph'})
AI: A StateGraph in LangGraph is the core class for building a stateful graph
application. It lets you define:
- State: the shared data held by the graph, often a TypedDict
- Nodes: functions that read and update that state
- Edges: connections between nodes, static or conditionally selected
- Compilation: validation and preparation of the graph for execution
...
In short: a StateGraph is a graph whose nodes communicate through a
persistent shared state.

You: What's the current weather in Tehran?
  [tool call] search_the_web({'query': 'current weather Tehran', 'time_range': 'day'})
AI: Here's the current weather in Tehran:
- Temperature: 29.8°C (85.6°F)
- Condition: Sunny
- Humidity: 7%
- Wind: 7.2 km/h from the WNW
It's a warm, sunny late afternoon in Tehran with very low humidity and calm winds.
```

This demonstrates correct tool routing: a LangGraph concept question triggered the document retriever (twice, refining its query on the first pass), while a live weather question correctly bypassed the document tool entirely in favor of web search — with no hardcoded logic telling it to do so.

## What I learned building this

- **Tool descriptions are the interface, not the code.** The LLM never inspects `tools.py` — it only ever sees each tool's name and docstring. Ambiguous or overlapping descriptions between `search_my_documents` and `search_the_web` were the single biggest lever over routing accuracy, more so than any graph logic.
- **A conditional edge plus a self-loop is what turns a chain into an agent.** Project 1's graph was a straight line (one node, one edge, done). Adding `add_conditional_edges` plus an edge from the tools node back to the agent node is the entire structural difference that enables multi-step, tool-using behavior.
- **Free-tier LLM routing introduces genuine reliability issues, not just quality issues.** Beyond occasional weak answers, `openrouter/free`'s random model selection produced outright authentication failures (`401 User not found`) that were transient and unrelated to my code or API key — confirmed by testing the same key directly via `curl`, which succeeded. This is a good example of infrastructure flakiness that a production system would need to handle with retries, not just prompt tuning.
- **Visibility into tool use matters for debugging and trust.** Adding a simple tool-call log (`[tool call] tool_name(args)`) made it possible to actually verify routing decisions were correct, rather than inferring them indirectly from the final answer's content.

## Known limitations

- No retry/fallback logic for transient API failures (auth errors, empty completions) — a failed call currently crashes the loop rather than retrying or degrading gracefully.
- Tool selection hasn't been stress-tested on genuinely ambiguous questions that plausibly need both tools in combination.
- Same model-consistency caveat as Project 1: `openrouter/free` may select a different underlying model per call, so response quality and tool-selection reliability can vary between runs.

## Next steps

- **Project 3** — multi-agent supervisor system: split this single agent into specialist sub-agents (e.g., researcher + writer) coordinated by a supervisor, plus reflection, structured output, human-in-the-loop approval, deployment, and evaluation.

## Credits

Built while working through *Building AI and LLM Applications with LangChain and LangGraph* by Mayo Oshin & Nuno Campos (O'Reilly).

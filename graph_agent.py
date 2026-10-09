import os
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
).bind_tools(tools)

from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by its order ID, e.g. ORD-1042."""
    return '{"status": "Shipped", "courier": "BlueDart", "eta": "Friday, 16 Oct"}'

tools = [get_order_status]
llm = ChatOllama(model="qwen2.5:7b").bind_tools(tools)

# Node 1: the model decides what to do
def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# Node 2: pause and ask a human before any tool runs
def approval(state: MessagesState) -> Command[Literal["tools", "__end__"]]:
    calls = [f"{c['name']}({c['args']})" for c in state["messages"][-1].tool_calls]
    decision = interrupt({"approve_these_calls": calls})
    if decision.strip().lower() == "yes":
        return Command(goto="tools")
    return Command(goto=END, update={"messages": [AIMessage(content="Cancelled by reviewer.")]})

# Draw the flowchart
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("approval", approval)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition, {"tools": "approval", END: END})
builder.add_edge("tools", "model")

graph = builder.compile(checkpointer=InMemorySaver())  # memory, so it can pause and resume
config = {"configurable": {"thread_id": "demo-1"}}

# Run until it pauses for approval
result = graph.invoke({"messages": [{"role": "user", "content": "Where is my order ORD-1042?"}]}, config)
print("\nPAUSED. Agent wants to run:", result["__interrupt__"][0].value)

# Resume with your decision
answer = input("Approve? (yes/no): ")
result = graph.invoke(Command(resume=answer), config)
print("\nFINAL:", result["messages"][-1].content)
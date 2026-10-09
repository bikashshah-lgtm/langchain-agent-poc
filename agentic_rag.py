import os, json
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Build the policy knowledge base once at startup (the RAG part) ---
text = open("policy.md").read()
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).create_documents([text])
store = InMemoryVectorStore.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text", base_url=BASE))

# --- Fake order database ---
ORDERS = {
    "ORD-1042": {"status": "Shipped", "courier": "BlueDart", "eta": "Friday, 16 Oct"},
    "ORD-2077": {"status": "Processing", "courier": None, "eta": "Not shipped yet"},
}

@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by its order ID, e.g. ORD-1042."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return json.dumps({"order_id": order_id, "error": "Order not found"})
    return json.dumps({"order_id": order_id, **order})

@tool
def search_policy(query: str) -> str:
    """Search the company policy on returns, refunds, shipping, cancellations and damaged items. Use this for any policy question."""
    hits = store.similarity_search(query, k=2)
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(hits))

agent = create_agent(
    ChatOllama(model="qwen2.5:7b", base_url=BASE, temperature=0),
    tools=[get_order_status, search_policy],
    system_prompt=(
        "You are a customer support assistant. Use get_order_status for order questions "
        "and search_policy for policy questions. Answer only from tool results, cite policy "
        "sources like [1], and say you don't know if the tools don't have the answer."
    ),
)

questions = [
    "Where is my order ORD-2077?",
    "Can I still cancel order ORD-2077 for free?",
    "What's the return window for electronics?",
]

for q in questions:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    tools_used = [m.name for m in result["messages"] if m.type == "tool"]
    print(f"\nQ: {q}\nTools used: {tools_used}\nA: {result['messages'][-1].content}")
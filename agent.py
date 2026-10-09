from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by its order ID, e.g. ORD-1042."""
    # Fake database result, same as your curl test
    return '{"status": "Shipped", "courier": "BlueDart", "eta": "Friday, 16 Oct"}'

import os
model = ChatOllama(
    model="qwen2.5:7b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
agent = create_agent(model, tools=[get_order_status])

result = agent.invoke({"messages": [{"role": "user", "content": "Check orders ORD-1042 and ORD-2077"}]})

# Print every step so you can see what LangChain did for you
for msg in result["messages"]:
    print(f"\n[{msg.type}]", msg.content or getattr(msg, "tool_calls", ""))
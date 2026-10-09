LangChain Agent PoC – Tool Calling, Human-in-the-Loop, RAG & Observability

A hands-on proof of concept for a customer-support AI agent, built step by step to explore the core building blocks of production GenAI applications. Everything runs locally on an open-source model (Qwen 2.5 7B via Ollama), so there are no API costs and no data leaves the machine.

Author: Bikash Kumar Shah · LinkedIn · Docker Hub: b1kash

What this repo demonstrates
File	Concept	What it shows
agent.py	Tool calling	A LangChain create_agent agent that decides when to call a get_order_status tool, including parallel tool calls for multiple orders
graph_agent.py	LangGraph + human-in-the-loop	The same agent built as an explicit graph, with an approval node that pauses (interrupt()) before any tool runs and resumes on a human "yes"/"no", using a checkpointer to save state
agent_langfuse.py	Observability (Langfuse)	The agent traced in Langfuse via a callback handler, alongside LangSmith tracing
rag.py	RAG	Chunking, local embeddings (nomic-embed-text), vector search and grounded answers with citations, plus a refusal test for out-of-scope questions
agentic_rag.py	Agentic RAG	One agent with two tools (order lookup + policy search) that chooses which to use and combines both to answer questions like "Can I still cancel this order for free?"
policy.md	Sample knowledge base	A short returns/refunds/shipping policy used by the RAG examples
Dockerfile	Containerisation	Packages the agents; secrets are passed at run time, never baked into the image
Architecture (agentic RAG)
User question
     │
     ▼
  Agent (Qwen 2.5 7B) ── decides which tool(s) to call
     │                       │
     ▼                       ▼
get_order_status        search_policy
(order "database")      (vector search over policy.md)
     │                       │
     └──────────┬────────────┘
                ▼
     Grounded answer with citations
                │
                ▼
   Traced in LangSmith / Langfuse
Tech stack
Frameworks: LangChain 1.x, LangGraph
Model: Qwen 2.5 7B (local, via Ollama) · Embeddings: nomic-embed-text
Vector store: LangChain in-memory vector store
Observability: LangSmith, Langfuse
Packaging: Docker (images on Docker Hub, tags 1.0–1.2)
Language: Python 3.12
Run it locally

Prerequisites: Python 3.10+ (tested on 3.12) and Ollama installed.

bash
# 1. Models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. Environment
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Run any example
python agent.py
python graph_agent.py      # type yes / no when prompted
python rag.py
python agentic_rag.py
Optional: tracing

Create a .env file (never commit it; it's in .gitignore). Don't put quotes around the values, because Docker's --env-file passes quotes through literally.

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=langchain-agent-poc
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
Run with Docker

The container talks to Ollama running on the host machine:

bash
docker run --rm --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  b1kash/langchain-agent-poc:1.2

# Human-in-the-loop version (interactive)
docker run -it --rm --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  b1kash/langchain-agent-poc:1.2 python graph_agent.py

Notes: images were built on Apple Silicon (ARM64). The published tags include the agent, LangGraph and Langfuse examples; rebuild the image to include the RAG scripts.

Key learnings
Tool descriptions drive agent behaviour. A vague description made the model ask for confirmation instead of calling the tool; a clear one fixed it.
LLM APIs are stateless. The application resends the full conversation each turn, so input tokens grow every step (seen in traces: 180 → 312 tokens).
Latency is dominated by model calls, not tools. Parallel tool calls saved a full model round trip.
Models are non-deterministic. The same question can produce different answers, which is why tracing and evaluation matter.
Tool outputs should be self-describing. Returning the order_id with each result stopped the model confusing two identical-looking responses.
Retrieval always returns something. The prompt's "say I don't know" instruction is what prevents hallucinated answers to out-of-scope questions.
Check citations, not just answers. The model once invented a source title, so tool results should carry real source labels.
Possible next steps
Return source metadata with each retrieved chunk for verifiable citations
Persistent vector store (e.g. pgvector) instead of in-memory
Automated evaluation set (faithfulness, relevance) run on every change
Expose the agent through a FastAPI endpoint

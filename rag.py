import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 1. Load and chunk the document
text = open("policy.md").read()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents([text], metadatas=[{"source": "policy.md"}])
print(f"Split into {len(chunks)} chunks")

# 2. Embed the chunks and store them
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=BASE)
store = InMemoryVectorStore.from_documents(chunks, embeddings)

llm = ChatOllama(model="qwen2.5:7b", base_url=BASE, temperature=0)

# 3. Retrieve, then generate
def ask(question):
    hits = store.similarity_search(question, k=2)
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(hits))
    prompt = f"""Answer using ONLY the context below. Cite sources like [1].
If the answer is not in the context, say "I don't know based on the policy."

Context:
{context}

Question: {question}"""
    print("\nQ:", question)
    print("Retrieved:", [h.page_content[:50] + "..." for h in hits])
    print("A:", llm.invoke(prompt).content)

ask("How many days do I have to return a phone?")
ask("How long does a refund take?")
ask("Do you ship to Mars?")
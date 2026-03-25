# -------------------------------------------------------------
# agent.py   –  LangChain Agent + Tavily search (Ollama model)
# -------------------------------------------------------------
import os
from typing import List
from dotenv import load_dotenv

from tavily import TavilyClient

# LangChain imports
from langchain_openai import ChatOpenAI          # OpenAI‑compatible wrapper
from langchain_core.tools import StructuredTool, Tool
from langchain.agents import create_agent

# --------------------------------------------------------------
# Helper to print the whole message chain
# --------------------------------------------------------------
def pretty_print(messages: List):
    for msg in messages:
        role = msg.type.upper()               # "HUMAN", "ASSISTANT", "TOOL"
        content = msg.content
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        print(f"{role}: {content}\n")

# ------------------------------------------------------------------
# 1️⃣ Load environment (Tavily API key)
# ------------------------------------------------------------------
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ------------------------------------------------------------------
# 2️⃣ Define the Tavily‑search tool
# ------------------------------------------------------------------
def _tavily_search(query: str) -> str:
    """
    Simple wrapper that returns a concise, human‑readable summary
    of the top 5 web results.
    """
    result = tavily.search(query, search_depth="basic", max_results=5)
    snippets: List[str] = []
    for i, doc in enumerate(result.get("results", []), start=1):
        title = doc.get("title", "")
        url   = doc.get("url", "")
        content = doc.get("content", "")
        # truncate each snippet to keep the total token count modest
        snippet = f"{i}. {title}\n{url}\n{content[:300]}..."
        snippets.append(snippet)
    return "\n".join(snippets) if snippets else "No results found."

# The `Tool` object tells LangChain how the LLM can call it.
tavily_tool = StructuredTool.from_function(_tavily_search)

# ------------------------------------------------------------------
# 3️⃣ Wrap Ollama as an OpenAI‑compatible chat model
# ------------------------------------------------------------------
# Ollama’s HTTP API mimics the OpenAI chat endpoint at /v1/chat/completions.
# We just point the client at the local server.
ollama_llm = ChatOpenAI(
    model="llama3.2:3b",  
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI‑compatible endpoint
    temperature=0.7,
    api_key="dummy",                 # Ollama ignores the key, but the field is required
)

# ------------------------------------------------------------------
# 4️⃣ Build the agent (function‑calling style)
# ------------------------------------------------------------------
# The `tools` argument can be a list of Tool / StructuredTool objects.
# LangChain will automatically add them to the system prompt and will
# parse the model's function‑call JSON.
agent = create_agent(
    model=ollama_llm,
    tools=[tavily_tool],
    # Optional: make the system prompt a bit more explicit
    system_prompt=(
        "You are a helpful assistant. Answer questions directly when you know the answer. "
        "If you need current or detailed information, call the `tavily_search` tool. "
        "Only call the tool when necessary."
        "in your answer say if you used the tool and what info you got from it. Be concise.Otherwise, just answer the question based on your existing knowledge and say that you didn't need to use the tool."
    ),
)

# ------------------------------------------------------------------
# 5️⃣ Simple REPL – same UX as before
# ------------------------------------------------------------------
def main() -> None:
    print("\n=== LangChain Agent (Ollama + Tavily) ===")
    print("Type 'exit' or press Ctrl‑C to quit.\n")
    while True:
        try:
            user_q = input("🧑‍💻 You: ").strip()
            if user_q.lower() in {"exit", "quit"}:
                print("👋 Bye!")

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_q}]},
                return_intermediate_steps=True,   # keep the full message list
            )
            pretty_print(result["messages"])
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as exc:
            print(f"\n❗ Error: {exc}\n")

if __name__ == "__main__":
    main()
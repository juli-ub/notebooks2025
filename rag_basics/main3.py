"""
***Coding assistants were used in the creation of the main f
ile and sub files***
"""

from document_loader import DataLoader
from text_splitter2 import TextSplitterUtil
from vector_store_util import VectorStoreUtil 
from rag_chain_util import GeminiRAGChain 

def main() -> None:
    loader = DataLoader(folder=".", filename="my_resume.txt")

    try:
        loader.load()
    except FileNotFoundError as exc:
        print(f"❌  {exc}")
        return

    # Print the file content
    # print("=== File content ===")
    # print(loader.content or "(empty)")

    #===========================================================
    # splitter = TextSplitter(chunk_size=200, overlap=40)
    # chunks = splitter.split(loader.content)

    # print(f"Split into {len(chunks)} chunks (chunk_size=200, overlap=40).")
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    #     print(chunk)
    #=========================================================

    docs = loader.content
    if docs is None:
        print("❌  No content loaded.")
        return

    text_splitter = TextSplitterUtil(chunk_size=250, chunk_overlap=50)
    all_splits = text_splitter.split_text(docs)

    #=========================================================
    # For demo purposes – just prints the first split
    # print("First split:")
    # print(all_splits[0] if all_splits else "(no splits)")   
    #=========================================================

    # embedder = EmbeddingUtil()          # uses OpenAIEmbeddings by default
    # embedder.add_texts(all_splits)      # builds FAISS index

    # # embedder.persist(pathlib.Path("faiss_index"))  # optional

    # # Demo query
    # query = "data engineering experience"
    # query_vec = embedder.embed([query])[0]
    # hits = embedder.search(query_vec, k=3)

    # print("\n🔍 Top 3 matches:")
    # for i, (text, score) in enumerate(hits, 1):
    #     print(f"{i}. [Score: {score:.4f}] {text[:120]}…")
    #=========================================
   
    
    # Initializes the VectorStoreUtil with your text splits
    vector_store_manager = VectorStoreUtil(splits=all_splits)
    
    # Creates the FAISS vector store
    vector_store = vector_store_manager.create_vector_store()


    #==========================================================
    # Optional: Perform a sample search
    # print("\nPerforming a sample similarity search...")
    # # query = "What skills are mentioned in the resume?"
    # # query = "Where did Julian go to school?"
    # # query = "what skill is mentioned the most often in the provided resume?"
    # results = vector_store_manager.search_store(query)

    # print(f"Found {len(results)} relevant results for query: '{query}'")
    # for i, doc in enumerate(results, 1):
    #     print(f"\n--- Result {i} ---")
    #     print(doc.page_content)
    #=================================================================
    
    # Define a retriever object from the vector store
    retriever = vector_store.as_retriever()

     # --- RAG Chain Logic ---
    rag_manager = GeminiRAGChain()
    # Passes the retriever to the chain setup
    rag_manager.setup_chain(retriever) 

    # --- Asks a question ---
    query = "What specific programming languages is the applicant proficient in?"

    final_answer = rag_manager.generate_response(query)

    print(f"\n--- Final Answer from Gemini for query: '{query}' ---")
    print(final_answer)

if __name__ == "__main__":
    main()




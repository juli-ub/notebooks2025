import os
import getpass
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

class VectorStoreUtil:
    def __init__(self, splits: list[str]):
        self.splits = splits
        load_dotenv() 
        api_key = os.getenv("GOOGLE_API_KEY")
        # Prompt user for API key if not set
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API key :\n")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        self.vector_store = None

    def create_vector_store(self) -> FAISS:
        """
        Embeds the text splits and creates a FAISS vector store.
        """
        print("Embedding documents and creating vector store with FAISS...")
        # Converts list of strings into a list of LangChain Document objects
        documents = [Document(page_content=t) for t in self.splits]
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print("Vector store created successfully.")
        return self.vector_store

    def search_store(self, query: str, k: int = 4):
        """
        Performs a similarity search on the created vector store.
        """
        if self.vector_store is None:
            print("Vector store has not been created yet. Call create_vector_store() first.")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
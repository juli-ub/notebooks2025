from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class GeminiRAGChain:
    def __init__(self):
        # Initialize the Gemini LLM (Gemini Pro is good for chat)
        # It automatically uses the GOOGLE_API_KEY environment variable you set earlier
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        self.rag_chain = None

    def setup_chain(self, retriever):
        """
        Sets up the LangChain RAG chain using LCEL.
        """
        # Define the prompt template
        template = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use only the provided context.

        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_response(self, question: str) -> str:
        """
        Generates a natural language response to a question using the RAG chain.
        """
        if self.rag_chain is None:
            return "RAG chain is not set up. Call setup_chain() first."

        print(f"\n--- Sending query to Gemini LLM ---")
        response = self.rag_chain.invoke(question)
        return response
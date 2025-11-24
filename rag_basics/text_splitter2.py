# text_splitter2.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitterUtil:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # The splitter has various delimiters, like single newlines, spaces, and empty strings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  
                "\n",   
                " ",     
                "",      
            ]
        )

    def split_text(self, text: str) -> list[str]:
        """
        Splits the given text into chunks using the configured text splitter.
        """
        return self.text_splitter.split_text(text)


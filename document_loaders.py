import os 
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader
)
from dotenv import load_dotenv


load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file.flush()
        temp_file_path = Path(temp_file.name)


        try:
            loader = TextLoader(temp_file_path)
            documents = loader.load()

            print(f"Loaded {len(documents)} document(s)")
            print(f"Document content: {documents[0].page_content[:100]}")  # Print the first 100 characters of the document content
            print(f"Document metadata: {documents[0].metadata}")
            # print the loaded docuemnts
            """
            for doc in documents:
                print("Document content:", doc.page_content)
                print(doc)
                print(f"Document content: {doc.page_content}")
                """
        finally:
            os.remove(temp_file_path)


def pdf_loader(pdf_path :str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from PDF")
    for doc in documents:
        print("Document content:", doc.page_content)
        print(doc)
        print(f"Document Metadata: {doc.metadata}")


if __name__ == "__main__":
    #load_text_file()
    pdf_loader("/Users/pushprajthakur/langc-course/docs/langchain_demo.pdf")
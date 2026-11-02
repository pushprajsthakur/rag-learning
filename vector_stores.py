from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil
from dotenv import load_dotenv


load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(
    model ="gemini-embedding-001"
)

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def chroma_basics():
    with tempfile.TemporaryDirectory() as temdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=temdir
        )
        print(
            f"Vector store created with {vectorstore._collection.count()} documents and persisted."
        )

        #perform similarity search
        #query = "What is LangChain?"
        query = "what is similarity search?"
        results = vectorstore.similarity_search(query,k=2)

        print(f"TOP 2 results for query: '{query}':")
        for i,doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source:{doc.metadata['source']})"
            )
def similarity_search_with_score():
    with tempfile.TemporaryDirectory() as temdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model,persist_directory=temdir
        )

        #perform similarity search with score
        query = "Explain vector stores"
        results_with_score = vectorstore.similarity_search_with_score(query,k=3)

        print(f"Top 3 results for query: '{query}':")
        for i,(doc,score) in enumerate(results_with_score):
            print(
                f"Result {i+1}: {doc.page_content} (Score: {score:.4f}, Source:{doc.metadata['source']})"
            )
def metadata_filtering():
    with tempfile.TemporaryDirectory() as temdir:
        #create a vecote store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,embedding=embeddings_model,persist_directory=temdir
        )
        
        query = "What databases are available?"

        #Without metadata filtering
        results_without_filter = vectorstore.similarity_search(query,k=5)
        print(f"Results without metadata filtering for query: '{query}':")
        for i,doc in enumerate(results_without_filter):
            print(
                f"Result {i+1}: {doc.page_content} (Source:{doc.metadata['source']})"
            )
        # with metadata filtering
        metadata_filter = {"topic": "database"}
        results_with_filter = vectorstore.similarity_search(query,k=5,filter=metadata_filter)
        print(f"\nResults with metadata filtering for query: '{query}' and filter: {metadata_filter}:")
        for i,doc in enumerate(results_with_filter):
            print(
                f"Result {i+1}: {doc.page_content} (Source:{doc.metadata['source']})"
            )

if __name__ == "__main__":
    #chroma_basics()
    #similarity_search_with_score()
    metadata_filtering()
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """ Combine multiple retriever using weighted reciprocal rank fusion."""
    doc_score = {}  # page_document ->{score,doc}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank +rrf_k))
            score, _ = doc_score.get(key, (0.0,doc))
            doc_score[key] = (score + rrf_score, doc)

    sorted_docs = sorted(doc_score.values(), key=lambda x:x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]

class HybridRetriever:
    """ Production hybrid retriever with BM25 + Vector search"""

    def __init__(self, documents: List[Document], bm25_weight: float = 0.5, k: int =4):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # Intialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        # create vector store
        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )

        # create vector retriever
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query:str):
        """ Run hybrid search using weighted RRF"""
        return hybrid_retrieve(
            query,
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k
        )

    def add_docuemnts(self, documents: List[Document]):
        """ Add new document to bith retrievers"""
        # add to vector store
        self.vectorstore.add_documents(documents)

        # Recreate BM25 ( it does not support increment adds that is why for prod use S3 or other lexical search)
        all_docs = self.vectorstore.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc) for doc in all_docs["documents"]],
            k =self.k
        )

# Usage
retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-7742X specifiations")

for doc in results:
    print(doc.page_content[:100])






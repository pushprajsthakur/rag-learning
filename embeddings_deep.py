from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(
    model ="gemini-embedding-001"
)

def basic_embeddings():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings_model.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norms: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]
    batch_embeddings = embeddings_model.embed_documents(text)
    for i,emb in enumerate(batch_embeddings):
        print(f"Text {i+1}: - vector dimensions: {len(emb)}")
        print(f"Text {i+1}: First values: {emb[:5]}")
        print(f"Text {i+1}: Vecotor Norm: {np.linalg.norm(emb):.4f}")
        
def similarity_search():

    # Document sample for vector store
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = " What programming language exists?"
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    #compute cosine similarity between query and document vectors
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities),key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("Ranked documents by similarity:")
    for doc, score in ranked_docs:
        print(f" {score:.4f}: {doc}")

def embedded_caching():
    import tempfile
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore
    import numpy as np

    with tempfile.TemporaryDirectory() as temdir:
        store = LocalFileStore(root_path=temdir)

        cached_embeddings =CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings_model,
            document_embedding_cache=store,
            namespace="gemini-embedding-001",
            key_encoder="sha256"
        )

        text = "What is Reinforcement Learning?"

        print("First API call to embed the text...")
        vector1 = cached_embeddings.embed_documents([text])
        print(f" Embedded {len(vector1)} Documents")

        print("Second API call to embed the same text...")
        vector2 = cached_embeddings.embed_documents([text])
        print(f" Embedded {len(vector2)} Documents")

        print("\nVerifying that the cached embeddings are identical to the original embeddings...")
        print(f"{np.allclose(vector1[0], vector2[0])}")  # Should print True if they are identical


if __name__ == "__main__":
   # basic_embeddings()
   # batch_embeddings()
   # similarity_search()
   embedded_caching()
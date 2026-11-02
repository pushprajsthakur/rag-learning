"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
import tempfile
from dotenv import load_dotenv


load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(
    model ="gemini-embedding-001"
)

def create_kb():
    """ create a knowledge base from a list of documents """

    # split the documents into chunks
    splitter = Recursive
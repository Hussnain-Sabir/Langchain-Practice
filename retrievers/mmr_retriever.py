# maximum margin relevance
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

docs = [
    Document(page_content="Python is a high-level programming language used for web development and AI."),
    Document(page_content="Python is an extremely popular coding language for data science and web apps."),
    Document(page_content="Python syntax is clean, simple, and easy to learn for beginners."),
    Document(page_content="Java is a class-based, object-oriented programming language."),
    Document(page_content="JavaScript is the core language used to build interactive websites and web apps.")
]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_mmr_db"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

retrieved_docs = retriever.invoke("Tell me about programming languages")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"Retrieved {i}: {doc.page_content}")
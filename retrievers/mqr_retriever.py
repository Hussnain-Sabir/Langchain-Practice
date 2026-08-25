# multiple query retriever
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.retrievers import MultiQueryRetriever
from langchain_core.documents import Document

load_dotenv(find_dotenv())

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

docs = [
    Document(page_content="To reduce vehicle engine noise, check for loose belts or low motor oil."),
    Document(page_content="Regular maintenance keeps your car running smoothly and quietly."),
    Document(page_content="Screeching sounds often indicate worn-out brake pads that need replacement.")
]

vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="qwen2.5:3b")

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

query = "Why is my car making loud sounds?"
retrieved_docs = multiquery_retriever.invoke(query)

print(f"Retrieved {len(retrieved_docs)} unique documents:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"Doc {i}: {doc.page_content}")
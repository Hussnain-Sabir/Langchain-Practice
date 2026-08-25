#conceptual compressor retriever
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

docs = [
    Document(
        page_content=(
            "The BMW S1000RR features a 999cc inline-four engine producing 205 horsepower. "
            "It has a top speed of over 185 mph and weighs roughly 434 lbs. "
            "The company was originally founded in 1916 as a manufacturer of aircraft engines."
        )
    ),
    Document(
        page_content=(
            "Pizza is a traditional Italian dish consisting of a flat base of leavened wheat-based dough. "
            "It is topped with tomatoes, cheese, and various other ingredients. "
            "The world's largest pizza was made in Rome in 2012 and covered 13,580 square feet."
        )
    )
]

vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 1})

llm = ChatOllama(model="qwen2.5:3b")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
query = "How much power does the BMW S1000RR have?"
compressed_docs = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_docs, 1):
    print(f"Compressed Result {i}:\n{doc.page_content}")
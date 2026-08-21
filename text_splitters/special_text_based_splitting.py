from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

text="""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

loader= PyPDFLoader("../document_loader/sample.pdf")
doc= loader.load()

splitter= RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

chunk= splitter.split_documents(doc)

print(chunk[0].page_content)
print(len(chunk))
"""

splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size= 200,
    chunk_overlap=0,
)

chunk= splitter.split_text(text)

print(chunk[0])
print(len(chunk))
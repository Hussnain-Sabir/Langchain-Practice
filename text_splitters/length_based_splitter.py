from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

loader= PyPDFLoader("../document_loader/sample.pdf")

document= loader.load()

splitter= CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="")
result= splitter.split_documents(document)

print(result[0].page_content)
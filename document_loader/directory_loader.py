from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


loader= DirectoryLoader(
    path="document_loader",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs= loader.load()

print(docs[0].page_content)
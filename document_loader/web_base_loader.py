from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

url="https://python.langchain.com/v0.2/docs/introduction/"

loader= WebBaseLoader(url)

docs=loader.load()

print(docs[0].page_content)
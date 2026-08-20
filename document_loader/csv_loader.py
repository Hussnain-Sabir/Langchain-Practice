from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

loader= CSVLoader(file_path="sample.csv")

doc=loader.load()

print(doc[0].page_content)
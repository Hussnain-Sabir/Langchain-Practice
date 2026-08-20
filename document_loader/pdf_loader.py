from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
parser= StrOutputParser()

pdf= PdfReader("sample.pdf")
text= pdf.pages[0].extract_text()

document= Document(
    page_content= text,
    metadata= {"source" : "sample.pdf", "author" : "Hussnain Sabir"},
)

print(document.metadata)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
parser= StrOutputParser()

with open("hello.txt", "r") as f:
    doc= f.read()

document= Document(
    page_content=doc,
    meta_data={"source" : "hello.txt", "author" : "Hussnain Sabir"}
)

prompt= ChatPromptTemplate([
    ("system" , "You are an expert text explainer."),
    ("human"  , "Explain this the txt file in short\n {file}")
])

chain= prompt | model | parser

result= chain.invoke({"file" : document.page_content})
print(result)

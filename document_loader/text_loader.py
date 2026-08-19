from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
parser= StrOutputParser()

loader= TextLoader("hello.txt")
doc= loader.load()

prompt= ChatPromptTemplate([
    ("system" , "You are an expert text explainer."),
    ("human"  , "Explain this the txt file in short\n {file}")
])

chain= prompt | model | parser

result= chain.invoke({"file" : doc[0].page_content})
print(result)

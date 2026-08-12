from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

template1= ChatPromptTemplate([
    ("system" , "You are an expert Report Generator."),
    ("human"  , "Generate a report on this {topic}.")
])

template2= ChatPromptTemplate([
    ("system" , "You are an expert Report Summarizer."),
    ("human"  , "Generate a 3 line summary of this {report}.")
])

parser= StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result= chain.invoke({"topic" : "How big is the universe"})

print(result)


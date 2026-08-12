from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")


prompt= ChatPromptTemplate([
        ("system" , "You are an expert car expert."),
        ("human"  , "Give me info about this {car}"),
])


parser= StrOutputParser()

chain= prompt | model | parser

result= chain.invoke({"car" : "Lamborghini"})

print(result)


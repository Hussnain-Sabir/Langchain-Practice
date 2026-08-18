from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser= StrOutputParser()

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert idea generator."),
    ("human"  , "Generate an 2 line idea on this {topic}.")
])

prompt2= ChatPromptTemplate([
    ("system" , "You are an expert Motivational Speaker."),
    ("human" , "Motivate me in 2 lines to do this {topic}.")
])

chain= RunnableParallel({
    "idea" : prompt1 | model | parser,
    "motivation" : prompt2 | model | parser
})


result= chain.invoke({"topic" : "E-commerce"})

print(result)
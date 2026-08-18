from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser= StrOutputParser()

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert joke makker."),
    ("human"  , "Give an amazing short joke on {topic}.")
])

prompt2= ChatPromptTemplate([
    ("system" , "You are an expert joke explainer"),
    ("human"  , "Explain this joke in easy words \n {joke}.")
])

chain= RunnableSequence(prompt1, model, parser, prompt2, model, parser)
# chain = prompt1 | model | parser | prompt2 | model | parser

result= chain.invoke({"topic" : "foods"})

print(result)

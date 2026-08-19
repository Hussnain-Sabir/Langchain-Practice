from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser= StrOutputParser()

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert poet."),
    ("human"  , "Make an intresting 2 line poetry on {topic}.")
])

prompt2= ChatPromptTemplate([
    ("system" , "You are expert poetry explainer."),
    ("human"  , "Explain this poetry in 3 lines and easy words\n {poetry}.")
])

chain1= prompt1 | model | parser

Parallel_chain= RunnableParallel({
    "poetry" : RunnablePassthrough(),
    "explaination" : prompt2 | model | parser
})

chain= chain1 | Parallel_chain

result= chain.invoke({"topic" : "Night Sky"})

print(result["poetry"])
print(result["explaination"])
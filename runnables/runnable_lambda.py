from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
parser= StrOutputParser()

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert joke maker."),
    ("human"  , "Make a funny joke on {topic}.")
])

joke_gen_chain= prompt1 | model | parser

parallel_chain= RunnableParallel({
    "joke" : RunnablePassthrough(),
    "length" : RunnableLambda(lambda x : len(x.split()))
})

chain= joke_gen_chain | parallel_chain

result= chain.invoke({"topic" : "Books"})

print(result["joke"])
print(result["length"])
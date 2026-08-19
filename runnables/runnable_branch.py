from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
parser= StrOutputParser()

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert chat report generator."),
    ("human"  , "Generate a report on this \n{response}")
])

prompt2= ChatPromptTemplate([
    ("system" , "You are an expert report summarizer."),
    ("human"  , "Summarize this report in short \n {report}")
])

report_chain= prompt1 | model | parser

conditional_chain= RunnableBranch(
    (lambda x : len(x.split())>300 , prompt2 | model | parser),
    RunnablePassthrough()
)

chain= report_chain | conditional_chain

result= chain.invoke({"response" : "Ben 10 omnitrix"})

print(result)
print(len(result.split()))
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

Gemini= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
Qwen_3b= ChatOllama(model="qwen2.5:3b", temperature=0.5)
llm= HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation", temperature=0.5) 
Qwen_72b = ChatHuggingFace(llm=llm)


template1= ChatPromptTemplate([
    ("system" , "You are an expert notes creator."),
    ("human"  , "Generate short simple notes from the following text\n {text}.")
])

template2= ChatPromptTemplate([
    ("system" , "You are an expert quiz generator."),
    ("human"  , "Generate 5 short questions from the following text]\n {text}")
])

template3= ChatPromptTemplate([
    ("system" , "You are an expert document compiler"),
    ("human"  , "Merget the provided notes and quiz into a single document\n notes-> {notes} and {quiz}")
])

parser= StrOutputParser()

parallel_chain= RunnableParallel({
    "notes" : template1 | Gemini  | parser,
    "quiz"  : template2 | Qwen_3b | parser
})

merge_chain= template3 | Qwen_72b | parser

chain= parallel_chain | merge_chain


text= """
The Omnimatrix, better known as the Omnitrix, was a watch-like device that attached
to Ben Tennyson's wrist at the beginning of the series and is the device that the franchise revolves
around. Considered the most powerful technological weapon in the universe, the device is a portable
library of intergalactic genetic data that allowed the wielder to alter their DNA at will and rapidly
transform into a variety of different alien species, each with their own unique abilities.
"""

result= chain.invoke({"text" : text})
print(result)


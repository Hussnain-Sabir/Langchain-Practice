from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@tool
def multipy(a:int, b:int) -> int:
    """Multiply two integers together and return the result."""
    return a*b


llm= ChatGoogleGenerativeAI(model="gemini-3.5-flash")

model= llm.bind_tools([multipy])

result= model.invoke("Multiply 5 and 2.")

print(result)


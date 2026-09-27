from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.6-flash")

@tool
def search(query:str) ->str:
    """Search the web for a query and return results."""
    search=DuckDuckGoSearchRun()
    return search.invoke(query)

tools= [search]

agent=create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant that answers questions using the tools provided."
)


result= agent.invoke({
    "messages": [("human", "Search who is the founder of pakistan.")]
})
print(result)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
import requests
import json

load_dotenv(find_dotenv())


@tool
def get_conversion_factor(base_currency: str, target_currency:str) -> float:

    """ This function fetches the currency conversion factor between a given base currency and a target currency. """

    url= f"https://v6.exchangerate-api.com/v6/15cd4af0352d53f2eae69933/pair/{base_currency}/{target_currency}"

    response= requests.get(url)

    return response.json()



@tool
def convert(base_currency:int, conversion_rate:Annotated[float, InjectedToolArg]) -> float:
    """ Given a currency conversion rate this function calculate the target currency value from a given base currency value """

    return base_currency*conversion_rate


#ran out of gemini tokens, now using a local model
llm = ChatOllama(model="qwen2.5:3b")

model= llm.bind_tools([get_conversion_factor,convert])

query= HumanMessage("What is the conversion factor between USD and PKR, and based on that can you convert 10 USD to PKR.")

messages= [query]

ai_message= model.invoke(messages)
messages.append(ai_message)


for tool_call in ai_message.tool_calls:

    if tool_call['name']=="get_conversion_factor":

        tool_message1= get_conversion_factor.invoke(tool_call)
        conversion_rate= json.loads(tool_message1.content)["conversion_rate"]
        messages.append(tool_message1)

    if tool_call['name']=="conversion":

        tool_call['args']['converison_rate']= conversion_rate
        tool_message2=convert.invoke(tool_call)
        messages.append(tool_message2)


result=model.invoke(messages)

print(result.content)



from typing import TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class CarInfo(TypedDict):
    car_name: str
    top_speed_kmh: int
    is_electric: bool

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

structured_llm = llm.with_structured_output(CarInfo)

result = structured_llm.invoke("Tell me about the Tesla Model S Plaid.")

print(type(result)) 
print(result)


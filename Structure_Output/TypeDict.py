from typing import TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 1. Define the schema using TypedDict
class CarInfo(TypedDict):
    car_name: str
    top_speed_kmh: int
    is_electric: bool

# 2. Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Attach schema to model using .with_structured_output()
structured_llm = llm.with_structured_output(CarInfo)

# 4. Invoke
result = structured_llm.invoke("Tell me about the Tesla Model S Plaid.")

print(type(result)) # <class 'dict'>
print(result)
# Output: {'car_name': 'Tesla Model S Plaid', 'top_speed_kmh': 322, 'is_electric': True}


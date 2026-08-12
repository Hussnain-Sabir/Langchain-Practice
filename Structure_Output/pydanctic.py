from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 1. Define the Pydantic Model with descriptions and constraints
class CarDetails(BaseModel):
    car_name: str = Field(description="The full brand and model name of the car")
    top_speed_kmh: int = Field(description="Top speed converted to kilometers per hour")
    highlights: list[str] = Field(description="List of top 3 key features of the car")

# 2. Initialize Model & Enforce Structure
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm.with_structured_output(CarDetails)

# 3. Invoke
result = structured_llm.invoke("Give me details on the Bugatti Chiron.")

print(type(result)) # <class '__main__.CarDetails'>
print(result.car_name)       # Access with dot notation!
print(result.top_speed_kmh)  # 400
print(result.highlights)     # ['Fastest production car', ...]
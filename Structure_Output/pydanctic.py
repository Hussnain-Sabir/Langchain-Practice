from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class CarDetails(BaseModel):
    car_name: str = Field(description="The full brand and model name of the car")
    top_speed_kmh: int = Field(description="Top speed converted to kilometers per hour")
    highlights: list[str] = Field(description="List of top 3 key features of the car")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
structured_llm = llm.with_structured_output(CarDetails)

result = structured_llm.invoke("Give me details on the Bugatti Chiron.")

print(type(result))
print(result.car_name)      
print(result.top_speed_kmh)  
print(result.highlights)   
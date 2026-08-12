from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv())

# 1. Define the JSON structure using Pydantic
class MovieDetails(BaseModel):
    title: str = Field(description="The title of the movie")
    release_year: int = Field(description="Year the movie was released")
    genres: list[str] = Field(description="List of genres")
    rating_out_of_10: float = Field(description="IMDb rating")

# 2. Initialize model and attach the structured output
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
json_model = model.with_structured_output(MovieDetails)

# 3. Invoke directly
result = json_model.invoke("Give me details about the movie Inception.")

# Result is a Pydantic object
print(f"Title: {result.title}")
print(f"Year: {result.release_year}")

# 4. Convert Pydantic object to a real JSON string
json_string = result.model_dump_json(indent=2)
print("\n--- Raw JSON String ---")
print(json_string)
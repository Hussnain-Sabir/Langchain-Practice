from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Extract information from the text into a JSON object "
        "containing the keys: 'car_name', 'top_speed', and 'country_of_origin'.\n\n"
        "{format_instructions}"
    ),
    ("human", "{text}")
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chain = prompt | model | parser

result = chain.invoke({
    "text": "The Bugatti Chiron is manufactured in France and can hit a top speed of 420 km/h.",
    "format_instructions": parser.get_format_instructions()
})

print("Python Type:", type(result))
print("\nParsed Result:", result)
print("\nAccess key directly:", result.get("car_name"))
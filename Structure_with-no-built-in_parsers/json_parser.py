from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 1. Create a generic JsonOutputParser (No Pydantic schema passed!)
parser = JsonOutputParser()

# 2. Define your prompt, injecting format_instructions
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Extract information from the text into a JSON object "
        "containing the keys: 'car_name', 'top_speed', and 'country_of_origin'.\n\n"
        "{format_instructions}"
    ),
    ("human", "{text}")
])

# 3. Initialize Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 4. Chain: Prompt -> Model -> JsonOutputParser
chain = prompt | model | parser

# 5. Invoke the chain
result = chain.invoke({
    "text": "The Bugatti Chiron is manufactured in France and can hit a top speed of 420 km/h.",
    "format_instructions": parser.get_format_instructions()
})

# Result is automatically a clean Python dictionary
print("Python Type:", type(result))  # <class 'dict'>
print("\nParsed Result:", result)
print("\nAccess key directly:", result.get("car_name"))
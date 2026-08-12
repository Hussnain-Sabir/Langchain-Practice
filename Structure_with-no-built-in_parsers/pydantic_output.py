from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")

class Person(BaseModel):
    name : str= Field(description="Name of the person")
    age : int= Field(gt=18,description="Age of the person")
    city : str= Field(description="Name of the city, where the person belongs to")


parser= PydanticOutputParser(pydantic_object=Person)

template= ChatPromptTemplate([
    ("system" , "You are an data entery person, Extract or generate the requested data.\n{format_instructions}"),
    ("human" , "Generate the name, age and city of the fictional {place} person."),
])

chain= template | model | parser

result= chain.invoke({
    "place" : "London",
    "format_instructions" : parser.get_format_instructions(),
})

print(result)
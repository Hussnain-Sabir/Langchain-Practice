from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

response_schemas = [
    ResponseSchema(
        name="topic",
        description="The main subject of the input text."
    ),
    ResponseSchema(
        name="sentiment",
        description="Sentiment of the text (Positive, Negative, or Neutral)."
    ),
    ResponseSchema(
        name="key_points",
        description="A list of 2-3 key takeaways extracted from the text.",
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert content analyzer.\n{format_instructions}"
    ),
    ("human", "Analyze the following review:\n{user_input}")
])

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chain = prompt | model | parser

text= (
    "I bought the new headphones yesterday. The noise cancellation is unbelievable, "
    "and the battery lasts forever, but the ear cushions feel a bit tight after two hours."
)

result = chain.invoke({
    "user_input": text,
    "format_instructions": parser.get_format_instructions()
})

print("Python Type:", type(result))
print("\nParsed Output:")
print(result)

print("\nTopic:", result["topic"])
print("Sentiment:", result["sentiment"])
print("Key Points:", result["key_points"])
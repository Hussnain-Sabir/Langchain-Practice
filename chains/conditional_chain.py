from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= ChatGoogleGenerativeAI(model="gemini-3.5-flash")

class Feedback(BaseModel):
    sentiment : Literal["positive","negative"]= Field(description="Give the sentiment of the feedback")

parser= StrOutputParser()
parser2= PydanticOutputParser(pydantic_object=Feedback)

prompt1= ChatPromptTemplate([
    ("system" , "You are an expert sentiment identifier. {format_instructions}"),
    ("human"  , "Classify the sentiment of the following feedback text into positive or negative\n {feedback}.")
])

classifier_chain= prompt1 | model | parser2

prompt2= ChatPromptTemplate([
    ("system", "You are an expert positive response writer."),
    ("human" , "Write an appropriate response to this positive feedback for the customer. Write exactly ONE response.\n {feedback}")
])

prompt3= ChatPromptTemplate([
    ("system", "You are an expert negative response writer."),
    ("human" , "Write an appropriate response to this negative feedback for the customer. Write exactly ONE response.\n {feedback}")
])

branch_chain = RunnableBranch(
    (lambda x: x["res"].sentiment == "positive", prompt2 | model | parser),
    (lambda x: x["res"].sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = RunnableLambda(
    lambda x: {"feedback": x["feedback"], "res": classifier_chain.invoke(x)}
) | branch_chain

result= chain.invoke({
    "feedback" : "This is the worst phone!",
    "format_instructions" : parser2.get_format_instructions(),
})
print(result)

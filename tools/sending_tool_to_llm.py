from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


@tool
def multipy(a: int, b: int) -> int:
    """Multiply two integers together and return the result."""
    return a * b


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

model = llm.bind_tools([multipy])

query = HumanMessage("Multiply 5 and 3")
messages = [query]

ai_msg = model.invoke(messages)
messages.append(ai_msg)

tool_result = multipy.invoke(ai_msg.tool_calls[0])
messages.append(tool_result)
final_result = model.invoke(messages).content

print(final_result)
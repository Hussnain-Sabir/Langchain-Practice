from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.5-flash")
text_result = llm.invoke("What is 2 + 2?") 


chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
chat_result = chat_model.invoke("What is 2 + 2?") 
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# 1. Text Model (Legacy / Simple Text Completion)
llm = GoogleGenerativeAI(model="gemini-3.5-flash")
text_result = llm.invoke("What is 2 + 2?") 
# Output: '4' (Raw string)


# 2. Chat Model (Modern Standard for Gemini/GPT)
chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
chat_result = chat_model.invoke("What is 2 + 2?") 
# Output: AIMessage(content='4') -> Use chat_result.text to get string
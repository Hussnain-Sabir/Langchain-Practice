from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {role}."),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{user_question}")                        
])

result= prompt_template.invoke({
    "role": "Python Developer",
    "chat_history": [], 
    "user_question": "Explain lambda functions"
})
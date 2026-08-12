from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Building a complete Chat Prompt with history slot
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {role}."),
    MessagesPlaceholder(variable_name="chat_history"), # Past messages go here
    ("human", "{user_question}")                        # Current question
])

# Filling the dynamic variables
filled_prompt = prompt_template.invoke({
    "role": "Python Developer",
    "chat_history": [], # Empty list if no previous conversation
    "user_question": "Explain lambda functions"
})
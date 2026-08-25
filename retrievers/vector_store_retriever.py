from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model= GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

docs= [
    Document(
        page_content="BMW S1000RR is my favourite bike.",
        metadata={"category": "vehicle", "type": "bike"}
    ),
    Document(
        page_content="BMW M5 is my favourite car.",
        metadata={"category": "vehicle", "type": "car"}
    ),
    Document(
        page_content="I like to eat pizza, its my favourite food.",
        metadata={"category": "food", "type": "fast_food"}
    )
]
doc_ids = ["doc_1", "doc_2", "doc_3"]

vector_store = Chroma.from_documents(
    documents=docs,
    ids=doc_ids,
    embedding=model,
    persist_directory="./chroma_db"
)

retriever= vector_store.as_retriever(search_kwargs={"k":1})

query= "What do i like to eat?"
result= retriever.invoke(query)

for i, doc in enumerate(result):
    print(doc.page_content)
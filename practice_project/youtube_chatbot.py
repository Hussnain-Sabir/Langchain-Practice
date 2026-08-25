from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# loading the video
video_id= "QNec5cxSEXo"
try:
    api= YouTubeTranscriptApi()
    transcript_list= api.fetch(video_id, languages=["en"])
    transcript= " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video.")
    exit()
except Exception as e:
    print("Error fetching transcript: ",e)
    exit()

# text splitting
splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
chunks= splitter.create_documents([transcript])

# using embedding model
embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# making embeddings and storing in vector store (ram only)
vector_store= Chroma.from_documents(chunks, embeddings)

# using retriever
retriever= vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

# making a chatbot model
model= ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

# making prompt for chatbot
prompt= ChatPromptTemplate([
    ("system" , """You are a helpful assistant,
    answer only from the provided transcript context,
    and if the context is insufficient just give a short 
    respone on something like not discussed in the video."""),

    ("human") , """{context}
                Question: {question}"""
])

# user query
question= "How was ben was able to defeat the other Alien X?"

# fetching retrieved documents
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain= RunnableParallel({
    "context" : retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser= StrOutputParser()

chain= parallel_chain | prompt | model | parser

# invoking the chain for final answer
result= chain.invoke(question)

print(result)
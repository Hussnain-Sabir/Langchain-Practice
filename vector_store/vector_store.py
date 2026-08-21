from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv(find_dotenv())
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


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
    embedding=embedding_model,
    persist_directory="./chroma_db"
)


print("--- 1. INSERTING NEW DOCUMENT ---")
new_doc = Document(
    page_content="I love drinking fresh Mango Shake in summer.",
    metadata={"category": "drink", "type": "beverage"}
)
vector_store.add_documents(documents=[new_doc], ids=["doc_4"])
print("Added document doc_4 successfully!\n")

# =====================================================================
# UPDATE (Edit)
# =====================================================================
print("--- 2. UPDATING EXISTING DOCUMENT ---")
updated_doc = Document(
    page_content="I like to eat pepperoni pizza and burgers, they are my favorite foods.",
    metadata={"category": "food", "type": "fast_food"}
)
vector_store.update_documents(ids=["doc_3"], documents=[updated_doc])
print("Updated doc_3 successfully!\n")

# =====================================================================
# DELETE (Remove)
# =====================================================================
print("--- 3. DELETING A DOCUMENT ---")
vector_store.delete(ids=["doc_2"]) # Deletes the BMW M5 car doc
print("Deleted doc_2 (BMW M5) successfully!\n")

# =====================================================================
# SEARCH METHODS (Read & Query)
# =====================================================================

# Method A: Standard Similarity Search
print("--- 4A. SIMILARITY SEARCH ---")
query = "What do I like to eat or drink?"
results = vector_store.similarity_search(query, k=2)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content}")

# Method B: Search with Distance / Similarity Score
# Lower score in distance metric = Higher similarity!
print("\n--- 4B. SEARCH WITH SCORES ---")
results_with_score = vector_store.similarity_search_with_score("favorite bike", k=1)
for doc, score in results_with_score:
    print(f"Matched: '{doc.page_content}' | Distance Score: {score:.4f}")

# Method C: Search with Metadata Filtering (Where clause)
print("\n--- 4C. SEARCH WITH METADATA FILTER ---")
filtered_results = vector_store.similarity_search(
    query="vehicles",
    k=2,
    filter={"category": "vehicle"} # Only searches documents marked as 'vehicle'
)
for doc in filtered_results:
    print(f"Filtered Result: {doc.page_content}")

# Method D: MMR (Maximal Marginal Relevance) Search
# Balances getting relevant results while avoiding near-identical duplicates
print("\n--- 4D. MMR SEARCH (DIVERSITY SEARCH) ---")
mmr_results = vector_store.max_marginal_relevance_search("food", k=2, fetch_k=5)
for doc in mmr_results:
    print(f"MMR Result: {doc.page_content}")

# =====================================================================
# CONVERT TO RETRIEVER (For RAG Chains)
# =====================================================================
print("\n--- 5. RETRIEVER INTERFACE ---")
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
retrieved_docs = retriever.invoke("What beverage do I like?")
print("Retrieved via LangChain chain component:", retrieved_docs[0].page_content)
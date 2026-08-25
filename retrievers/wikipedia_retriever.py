from langchain_community.retrievers import WikipediaRetriever
import wikipedia

wikipedia.set_user_agent("LangChainApp/1.0 (sunnyuar30@example.com)")

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "How old is ben 10, in ben 10 omniverse?"
result = retriever.invoke(query)

print(result[0].page_content)
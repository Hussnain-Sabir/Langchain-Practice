from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ShellTool
from langchain_community.utilities import WikipediaAPIWrapper


print("--- 1. DuckDuckGo Web Search Tool ---")
web_search_tool = DuckDuckGoSearchRun()
print("Tool Name:", web_search_tool.name)
print("Tool Description:", web_search_tool.description)
search_result = web_search_tool.invoke("LangChain latest updates")
print("\nSearch Output:\n", search_result[:300], "...\n")


print("--- 2. Shell Execution Tool ---")
shell_tool = ShellTool()
print("Tool Name:", shell_tool.name)
shell_result = shell_tool.invoke({"commands": ["echo 'Hello from LangChain Shell Tool!'", "whoami"]})
print("\nShell Output:\n", shell_result)


print("\n--- 3. Wikipedia Query Tool ---")
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_result = wiki_tool.invoke("Quantum Computing")
print("Wikipedia Output:\n", wiki_result)
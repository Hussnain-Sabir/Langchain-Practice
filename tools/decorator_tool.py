from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together and returns the result."""
    return a * b

result = multiply.invoke({"a": 4, "b": 5})
print("Result:", result)
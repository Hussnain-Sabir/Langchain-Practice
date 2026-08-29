from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 3, 'b': 3})

print(f"Result: {result}")
print(f"Name: {multiply_tool.name}")
print(f"Description: {multiply_tool.description}")
print(f"Args Schema: {multiply_tool.args}")
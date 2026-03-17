from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="sqlcoder", temperature=0)

EXPLAINER_PROMPT = PromptTemplate(
    input_variables=["sql"],
    template="""### Task
Explain the following SQL query in one simple, plain English sentence.

### SQL Query
{sql}

### Answer
Explanation:
"""
)

chain = EXPLAINER_PROMPT | llm

sql = "CREATE TABLE staff (first_name TEXT, last_name TEXT, dept TEXT, salary FLOAT);"

print("Generating Explanation...")
raw = chain.invoke({"sql": sql})
print("Result:")
print(repr(raw))

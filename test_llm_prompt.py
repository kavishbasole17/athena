from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="sqlcoder", temperature=0)

EXPLAINER_PROMPT = PromptTemplate(
    input_variables=["sql"],
    template="""You are a helpful assistant that explains SQL statements to non-technical users.

Given this SQL statement:
{sql}

Write ONE simple, plain English sentence that explains what this statement does, without using any technical jargon.
Output ONLY that single sentence, nothing else."""
)

chain = EXPLAINER_PROMPT | llm

sql = "CREATE TABLE staff (first_name TEXT, last_name TEXT, dept TEXT, salary FLOAT);"

print("Generating Explanation...")
raw = chain.invoke({"sql": sql})
print("Result:")
print(repr(raw))

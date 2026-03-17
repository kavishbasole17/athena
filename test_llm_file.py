from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="sqlcoder", temperature=0)

prompt = '''### Task
Generate a SQL query to answer [QUESTION]i need to create a table named staff and have the fields as first name, last name, dept, salary[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
CREATE TABLE dummy (id INT);

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]i need to create a table named staff and have the fields as first name, last name, dept, salary[/QUESTION]
'''

print("Invoking model...")
response = llm.invoke(prompt)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(response)
    
print("Done. Wrote to output.txt.")

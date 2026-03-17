from core.llm_engine import get_llm, build_sql_chain, build_explainer_chain, generate_sql, explain_sql

llm = get_llm("sqlcoder")
sql_chain = build_sql_chain(llm)
explainer_chain = build_explainer_chain(llm)

schema = "CREATE TABLE students (id INT, name VARCHAR);"
question = "i need to create a table named staff and have the fields as first name, last name, dept, salary"

print("Generating SQL...")
sql = generate_sql(sql_chain, schema, question)
print("FINAL SQL OUTPUT:")
print(repr(sql))

print("\nGenerating Explanation...")
explanation = explain_sql(explainer_chain, sql)
print("FINAL EXPLANATION OUTPUT:")
print(repr(explanation))

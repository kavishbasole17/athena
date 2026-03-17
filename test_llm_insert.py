from core.llm_engine import get_llm, build_sql_chain

llm = get_llm("sqlcoder")
sql_chain = build_sql_chain(llm)

schema = "CREATE TABLE food (name TEXT, price REAL);"
question = "add an entry into the table food. Shawarma is Rs. 100"

print("Generating SQL...")
sql = sql_chain.invoke({"schema": schema, "question": question})
print("RAW SQL OUTPUT:")
print(repr(sql))

question2 = "add entries into the food table which say - shawarma is Rs. 100, Momos are Rs. 60 and Frankie is Rs. 99"
print("Generating SQL 2...")
sql2 = sql_chain.invoke({"schema": schema, "question": question2})
print("RAW SQL OUTPUT 2:")
print(repr(sql2))

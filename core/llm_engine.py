"""
core/llm_engine.py
------------------
Handles:
  - Connection to local Ollama (sqlcoder model)
  - SQL Generation Chain: question -> schema -> SQL (supports ALL SQL types)
  - Explainer Chain: SQL -> plain English (with WARNING prefix for write/destructive ops)
"""

import logging
import re
import time
import sqlite3

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ─── Keywords that require HITL confirmation ──────────────────────────────────
# Any SQL that modifies state (data or schema) triggers the guardrail
WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|RENAME)\b",
    re.IGNORECASE,
)

# Subset that is specifically dangerous (shown with WARNING label)
DESTRUCTIVE_KEYWORDS = re.compile(
    r"\b(DELETE|DROP|TRUNCATE|ALTER)\b",
    re.IGNORECASE,
)

# ─── LLM Instance ─────────────────────────────────────────────────────────────

def get_llm(model: str = "sqlcoder") -> OllamaLLM:
    """Return a LangChain OllamaLLM instance."""
    return OllamaLLM(model=model, temperature=0)


# ─── SQL Generation Chain ─────────────────────────────────────────────────────

SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Rules
- Support ALL SQL statement types (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE, ALTER TABLE).
- Use standard SQLite syntax.

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
```sql
"""
)


def build_sql_chain(llm: OllamaLLM):
    """Return an LCEL chain that takes {schema, question} and outputs raw SQL."""
    return SQL_PROMPT | llm | StrOutputParser()


def generate_sql(sql_chain, schema: str, question: str) -> tuple[str, float]:
    """Run the SQL chain and return a cleaned SQL string and execution time."""
    start_time = time.perf_counter()
    try:
        raw_output = sql_chain.invoke({"schema": schema, "question": question})
        # Strip markdown fences if the model wraps output despite instructions
        sql = re.sub(r"```(?:sql)?", "", raw_output, flags=re.IGNORECASE).strip()
        sql = sql.strip("`").strip()
        elapsed_time = time.perf_counter() - start_time
        logger.info("SQL generated: %s", sql)
        return sql, elapsed_time
    except Exception as e:
        logger.error("SQL generation failed: %s", e)
        raise

# ─── Alternate SQL Generation Chain ───────────────────────────────────────────

ALT_SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Rules
- Support ALL SQL statement types.
- Use standard SQLite syntax strictly.

### Answer
Given the database schema, here is the plain SQL query (without markdown wrappers or explanations) that answers [QUESTION]{question}[/QUESTION]:
"""
)

def build_alt_sql_chain(llm: OllamaLLM):
    """Return an alternative LCEL chain for SQL generation."""
    return ALT_SQL_PROMPT | llm | StrOutputParser()

def generate_sql_alternate(alt_sql_chain, schema: str, question: str) -> tuple[str, float]:
    """Run the alternate SQL chain and return the SQL with elapsed time."""
    start_time = time.perf_counter()
    try:
        raw_output = alt_sql_chain.invoke({"schema": schema, "question": question})
        sql = raw_output.strip()
        # Even stricter fallback cleanup for Alternate Algorithm
        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = sql.strip("`").strip()
        elapsed_time = time.perf_counter() - start_time
        logger.info("Alternate SQL generated: %s", sql)
        return sql, elapsed_time
    except Exception as e:
        logger.error("Alternate SQL generation failed: %s", e)
        raise

# ─── Reliability Metric ───────────────────────────────────────────────────────

def validate_sql_syntax(sql: str) -> bool:
    """Check if the SQL syntax is valid against SQLite (dry run)."""
    sql_stripped = sql.strip()
    if not sql_stripped:
        logger.warning("SQL Validation Failed: empty SQL string")
        return False
    try:
        from core.db_manager import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        sql_upper = sql_stripped.upper()
        if any(sql_upper.startswith(kw) for kw in ("CREATE", "DROP", "ALTER")):
             # DDL EXPLAIN
             conn.execute(f"EXPLAIN {sql_stripped}")
        else:
             conn.execute(f"EXPLAIN {sql_stripped}")
        conn.close()
        return True
    except sqlite3.Error as e:
        logger.warning("SQL Validation Failed: %s", e)
        return False


# ─── Explainer Chain ──────────────────────────────────────────────────────────

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


def build_explainer_chain(llm: OllamaLLM):
    """Return an LCEL chain that takes {sql} and outputs a plain English explanation."""
    return EXPLAINER_PROMPT | llm | StrOutputParser()


def explain_sql(explainer_chain, sql: str) -> str:
    """Run the explainer chain and prepend WARNING if the SQL is destructive."""
    try:
        explanation = explainer_chain.invoke({"sql": sql}).strip()
        if DESTRUCTIVE_KEYWORDS.search(sql):
            explanation = f"⚠  WARNING — DESTRUCTIVE OPERATION: {explanation}"
        elif WRITE_KEYWORDS.search(sql):
            explanation = f"✎  WRITE OPERATION: {explanation}"
        logger.info("Explanation generated: %s", explanation)
        return explanation
    except Exception as e:
        logger.error("SQL explanation failed: %s", e)
        raise

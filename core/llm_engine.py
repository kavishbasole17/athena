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
    template="""You are an expert SQLite assistant. Given the database schema below, write a single valid SQL statement that satisfies the user's request.

### Database Schema:
{schema}

### User Request:
{question}

### Rules:
- Output ONLY the raw SQL statement, nothing else.
- Do NOT include explanations, markdown fences (```), or commentary.
- Support ALL SQL statement types:
    * SELECT  — for queries / lookups
    * INSERT  — to add new rows
    * UPDATE  — to modify existing rows
    * DELETE  — to remove rows
    * CREATE TABLE  — to create a new table (use IF NOT EXISTS)
    * DROP TABLE    — to delete a table entirely (use IF EXISTS)
    * ALTER TABLE   — to add/rename/drop columns
- Use standard SQLite syntax.
- For CREATE TABLE, always define primary keys and sensible column types.
- For INSERT, supply realistic placeholder values if none are given, unless the user provides them.

SQL:""",
)


def build_sql_chain(llm: OllamaLLM):
    """Return an LCEL chain that takes {schema, question} and outputs raw SQL."""
    return SQL_PROMPT | llm | StrOutputParser()


def generate_sql(sql_chain, schema: str, question: str) -> str:
    """Run the SQL chain and return a cleaned SQL string."""
    try:
        raw_output = sql_chain.invoke({"schema": schema, "question": question})
        # Strip markdown fences if the model wraps output despite instructions
        sql = re.sub(r"```(?:sql)?", "", raw_output, flags=re.IGNORECASE).strip()
        sql = sql.strip("`").strip()
        logger.info("SQL generated: %s", sql)
        return sql
    except Exception as e:
        logger.error("SQL generation failed: %s", e)
        raise


# ─── Explainer Chain ──────────────────────────────────────────────────────────

EXPLAINER_PROMPT = PromptTemplate(
    input_variables=["sql"],
    template="""You are a helpful assistant that explains SQL statements to non-technical users.

Given this SQL statement:
{sql}

Write ONE simple, plain English sentence that explains what this statement does, without using any technical jargon.
Output ONLY that single sentence, nothing else.""",
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

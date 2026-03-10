"""
core/db_manager.py
------------------
Handles:
  - SQLite database scaffolding (university.db)
  - Dynamic DDL extraction (all user tables)
  - ChromaDB vector store ingestion & retrieval
  - execute_sql: generic execution for ANY SQL statement type
"""

import sqlite3
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "university.db"
CHROMA_DIR = str(BASE_DIR / "chroma_db")

logger = logging.getLogger(__name__)


# ─── Database Setup ───────────────────────────────────────────────────────────

def setup_database() -> None:
    """Create university.db with students and courses tables, seeded with dummy data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS courses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                course_name TEXT    NOT NULL,
                credits     INTEGER NOT NULL,
                instructor  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS students (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL,
                age       INTEGER NOT NULL,
                gpa       REAL    NOT NULL,
                course_id INTEGER,
                FOREIGN KEY (course_id) REFERENCES courses(id)
            );
        """)

        # Seed courses only if empty
        cursor.execute("SELECT COUNT(*) FROM courses")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                "INSERT INTO courses (course_name, credits, instructor) VALUES (?, ?, ?)",
                [
                    ("Introduction to Computer Science", 3, "Dr. Alan Turing"),
                    ("Linear Algebra",                   4, "Dr. Emmy Noether"),
                    ("Data Structures",                  3, "Prof. Donald Knuth"),
                    ("Machine Learning",                 4, "Dr. Geoffrey Hinton"),
                    ("Database Systems",                 3, "Dr. Edgar Codd"),
                ],
            )

        # Seed students only if empty
        cursor.execute("SELECT COUNT(*) FROM students")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                "INSERT INTO students (name, age, gpa, course_id) VALUES (?, ?, ?, ?)",
                [
                    ("Alice Johnson",  20, 3.9,  1),
                    ("Bob Smith",      22, 3.2,  2),
                    ("Carol White",    21, 3.7,  3),
                    ("David Lee",      23, 2.8,  4),
                    ("Eva Martinez",   20, 3.85, 5),
                    ("Frank Brown",    24, 3.1,  1),
                    ("Grace Kim",      21, 3.95, 2),
                    ("Henry Davis",    22, 2.6,  3),
                    ("Iris Wilson",    20, 3.5,  4),
                    ("Jack Taylor",    23, 3.0,  5),
                ],
            )

        conn.commit()
        logger.info("Database setup complete: %s", DB_PATH)
    except sqlite3.Error as e:
        logger.error("Database setup failed: %s", e)
        raise
    finally:
        conn.close()


# ─── DDL Extraction ───────────────────────────────────────────────────────────

def get_ddl_statements() -> dict[str, str]:
    """Return a dict of {table_name: ddl_string} for all user tables."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        rows = cursor.fetchall()
        ddl_map = {name: sql for name, sql in rows if sql}
        logger.info("Extracted DDL for tables: %s", list(ddl_map.keys()))
        return ddl_map
    except sqlite3.Error as e:
        logger.error("DDL extraction failed: %s", e)
        raise
    finally:
        conn.close()


# ─── ChromaDB Vector Store ────────────────────────────────────────────────────

def setup_vector_store():
    """Ingest all table DDLs into a persistent ChromaDB collection and return it."""
    try:
        ddl_map = get_ddl_statements()

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Delete and recreate to ensure fresh schema on each run
        try:
            client.delete_collection("schema_store")
        except Exception:
            pass

        collection = client.create_collection(
            name="schema_store",
            embedding_function=embedding_fn,
        )

        _populate_collection(collection, ddl_map)
        return collection
    except Exception as e:
        logger.error("Vector store setup failed: %s", e)
        raise


def refresh_vector_store(collection) -> None:
    """
    Re-sync the ChromaDB collection with the current database schema.
    Call this after any DDL statement (CREATE TABLE, DROP TABLE, ALTER TABLE).
    """
    try:
        ddl_map = get_ddl_statements()

        # Remove stale entries no longer in the DB
        existing = collection.get()
        existing_ids = set(existing["ids"])
        current_ids = {f"table_{name}" for name in ddl_map}

        stale_ids = existing_ids - current_ids
        if stale_ids:
            collection.delete(ids=list(stale_ids))
            logger.info("Removed stale schema entries: %s", stale_ids)

        # Upsert current schema
        _populate_collection(collection, ddl_map)
        logger.info("Vector store refreshed. Tables: %s", list(ddl_map.keys()))
    except Exception as e:
        logger.error("Vector store refresh failed: %s", e)
        raise


def _populate_collection(collection, ddl_map: dict) -> None:
    """Upsert all DDL entries into the ChromaDB collection."""
    if not ddl_map:
        logger.warning("No user tables found in database — vector store will be empty.")
        return

    documents, metadatas, ids = [], [], []
    for table_name, ddl in ddl_map.items():
        doc = f"Table: {table_name}\n\n{ddl}"
        documents.append(doc)
        metadatas.append({"table": table_name})
        ids.append(f"table_{table_name}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("ChromaDB populated with %d table schema(s).", len(documents))


def retrieve_relevant_schema(collection, question: str, n_results: int = 3) -> str:
    """
    Query ChromaDB for the DDL most relevant to the user's question.
    n_results capped to however many documents exist to avoid Chroma errors.
    """
    try:
        total = collection.count()
        if total == 0:
            logger.warning("Vector store is empty — returning empty schema.")
            return "(No tables found in database)"
        effective_n = min(n_results, total)
        results = collection.query(query_texts=[question], n_results=effective_n)
        schemas = results["documents"][0]
        combined = "\n\n".join(schemas)
        logger.info("Retrieved %d schema(s) for question.", len(schemas))
        return combined
    except Exception as e:
        logger.error("Schema retrieval failed: %s", e)
        raise


# ─── SQL Execution ────────────────────────────────────────────────────────────

def execute_sql(sql: str) -> tuple[list, list]:
    """
    Execute ANY SQL statement against university.db.

    Returns:
        (columns, rows) for SELECT queries.
        ([], [(message,)]) for write operations, reporting rows affected.
        ([], [(message,)]) for DDL statements (CREATE / DROP / ALTER).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        # Enable foreign key enforcement
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # executescript is needed for multi-statement DDL; use it for DDL keywords
        sql_upper = sql.strip().upper()
        if any(sql_upper.startswith(kw) for kw in ("CREATE", "DROP", "ALTER")):
            # Multi-statement safe path
            conn.executescript(sql)
            conn.commit()
            logger.info("DDL executed successfully: %s", sql[:80])
            return [], [("DDL statement executed successfully.",)]
        else:
            cursor.execute(sql)
            if cursor.description:          # SELECT — fetch results
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()
                conn.commit()
                logger.info("SELECT executed. Rows returned: %d", len(rows))
                return columns, rows
            else:                           # INSERT / UPDATE / DELETE
                conn.commit()
                msg = f"{cursor.rowcount} row(s) affected."
                logger.info("DML executed. %s", msg)
                return [], [(msg,)]
    except sqlite3.Error as e:
        logger.error("SQL execution error: %s", e)
        raise
    finally:
        conn.close()

"""
main.py
-------
RAG Text-to-SQL Agent — Interactive CLI with Human-in-the-Loop (HITL) guardrail.

Supports ALL SQL operations:
  SELECT · INSERT · UPDATE · DELETE · CREATE TABLE · DROP TABLE · ALTER TABLE

Flow:
  1. User types a natural-language request
  2. Agent retrieves relevant DB schema from ChromaDB
  3. LLM generates the appropriate SQL statement
  4. LLM explains the SQL in plain English
     (⚠ WARNING prefix for destructive ops; ✎ WRITE label for other mutations)
  5. *** Execution pauses — user must type Y or N ***
  6. SQL is executed only on explicit 'Y' confirmation
  7. After DDL (CREATE/DROP/ALTER), the ChromaDB schema store is auto-refreshed
  8. All events logged to agent_audit.log
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# ─── Audit Logging Setup (must happen before any imports that use logging) ────
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "agent_audit.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Silence noisy third-party loggers
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("agent")

# ─── Local imports ────────────────────────────────────────────────────────────
from core.db_manager import (
    setup_database,
    setup_vector_store,
    retrieve_relevant_schema,
    refresh_vector_store,
    execute_sql,
)
from core.llm_engine import (
    get_llm,
    build_sql_chain,
    build_explainer_chain,
    generate_sql,
    explain_sql,
    WRITE_KEYWORDS,
    DESTRUCTIVE_KEYWORDS,
    build_alt_sql_chain,
    generate_sql_alternate,
    validate_sql_syntax,
)

import re

# DDL operations that change the table structure (need schema refresh afterwards)
DDL_KEYWORDS = re.compile(r"\b(CREATE|DROP|ALTER)\b", re.IGNORECASE)


# ─── UI Helpers ───────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║       RAG Text-to-SQL Agent  |  HITL Guardrail Active   ║
║       Model: sqlcoder (Ollama)   DB: university.db      ║
║       Supports: SELECT · INSERT · UPDATE · DELETE       ║
║                 CREATE · DROP · ALTER                   ║
╚══════════════════════════════════════════════════════════╝
"""


def print_box(title: str, content: str, width: int = 62) -> None:
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╠{border}╣")
    for line in content.strip().splitlines():
        # Wrap long lines
        while len(line) > width - 2:
            print(f"║  {line[:width - 2]}║")
            line = line[width - 2:]
        print(f"║  {line:<{width - 2}}║")
    print(f"╚{border}╝")


def display_results(columns: list, rows: list) -> None:
    if not rows:
        print("\n  (No rows returned.)")
        return
    if not columns:
        # Write-op or DDL message
        print(f"\n  ➜  {rows[0][0]}")
        return
    # Format as table
    col_widths = [
        max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
        for i, c in enumerate(columns)
    ]
    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(f"\n  {header}")
    print(f"  {separator}")
    for row in rows:
        print("  " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print(f"\n  ({len(rows)} row{'s' if len(rows) != 1 else ''} returned)")


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)
    logger.info("=" * 60)
    logger.info("Agent session started at %s", datetime.now().isoformat())

    # ── Initialise infrastructure ─────────────────────────────────────────────
    print("  [*] Setting up database...")
    try:
        setup_database()
        print("  [✓] university.db ready.\n")
    except Exception as e:
        logger.error("Fatal: database setup failed — %s", e)
        sys.exit(1)

    print("  [*] Loading ChromaDB vector store...")
    try:
        collection = setup_vector_store()
        print("  [✓] Schema vector store ready.\n")
    except Exception as e:
        logger.error("Fatal: vector store setup failed — %s", e)
        sys.exit(1)

    print("  [*] Connecting to Ollama (sqlcoder)...")
    try:
        llm = get_llm(model="sqlcoder")
        sql_chain = build_sql_chain(llm)
        alt_sql_chain = build_alt_sql_chain(llm)
        explainer_chain = build_explainer_chain(llm)
        print("  [✓] LLM chains ready.\n")
    except Exception as e:
        logger.error("Fatal: LLM initialisation failed — %s", e)
        sys.exit(1)

    print("  Ask anything in plain English — queries, inserts, deletes, table creation, etc.")
    print("  Type 'quit' or 'exit' to stop.\n")
    print("  " + "─" * 60)

    # ── Interactive loop ──────────────────────────────────────────────────────
    while True:
        try:
            question = input("\n  🔍 Your request: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  [!] Session interrupted. Goodbye!")
            logger.info("Session ended by user interrupt.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("\n  [!] Exiting agent. Goodbye!")
            logger.info("Session ended by user command.")
            break

        logger.info("USER REQUEST: %s", question)

        # Step 1 — Retrieve relevant schema
        try:
            print("\n  [*] Retrieving relevant schema from ChromaDB...")
            schema = retrieve_relevant_schema(collection, question)
        except Exception as e:
            print(f"\n  [✗] Schema retrieval error: {e}")
            logger.error("Schema retrieval error: %s", e)
            continue

        # Step 2 — Generate SQL using both algorithms
        try:
            print("  [*] Generating SQL via Algorithm 1 (Current)...")
            sql1, time1 = generate_sql(sql_chain, schema, question)
            is_reliable1 = validate_sql_syntax(sql1)
        except Exception as e:
            print(f"\n  [✗] Algorithm 1 SQL generation error: {e}")
            logger.error("SQL generation error: %s", e)
            continue
            
        try:
            print("  [*] Generating SQL via Algorithm 2 (Alternate)...")
            sql2, time2 = generate_sql_alternate(alt_sql_chain, schema, question)
            is_reliable2 = validate_sql_syntax(sql2)
        except Exception as e:
            print(f"\n  [✗] Algorithm 2 SQL generation error: {e}")
            logger.error("Alt SQL generation error: %s", e)
            continue

        print("\n  [Metrics Comparison]")
        print(f"  Algorithm 1: {time1:.2f}s | Reliable SQL: {'Yes' if is_reliable1 else 'No'} | SQL: {sql1}")
        print(f"  Algorithm 2: {time2:.2f}s | Reliable SQL: {'Yes' if is_reliable2 else 'No'} | SQL: {sql2}")

        # Choose the best SQL to proceed
        if is_reliable1 and not is_reliable2:
            sql = sql1
            print(f"\n  [*] Proceeding with Algorithm 1 output.")
        elif is_reliable2 and not is_reliable1:
            sql = sql2
            print(f"\n  [*] Proceeding with Algorithm 2 output.")
        else:
            # If both are reliable, or both are unreliable, default to Algo 1.
            sql = sql1
            print(f"\n  [*] Proceeding with Algorithm 1 output as default.")

        # Step 3 — Explain SQL
        try:
            print("  [*] Generating plain-English explanation...")
            explanation = explain_sql(explainer_chain, sql)
        except Exception as e:
            explanation = "(Explanation unavailable)"
            logger.warning("Explanation generation failed: %s", e)

        # Step 4 — Display to user
        print_box("⚡ Generated SQL", sql)
        print_box("💬 What this does", explanation)

        # ── HITL GUARDRAIL ────────────────────────────────────────────────────
        # Always confirm for any state-changing operation; reads go straight through
        is_write = bool(WRITE_KEYWORDS.search(sql))

        if is_write:
            print()
            print("  ┌──────────────────────────────────────────────────┐")
            print("  │  ⚠  HUMAN-IN-THE-LOOP CONFIRMATION REQUIRED      │")
            print("  └──────────────────────────────────────────────────┘")

            while True:
                try:
                    confirm = input("  Execute this SQL against the database? [Y/N]: ").strip().upper()
                except (KeyboardInterrupt, EOFError):
                    confirm = "N"
                    print()

                if confirm in {"Y", "N"}:
                    break
                print("  [!] Please type Y to execute or N to cancel.")
        else:
            # SELECT queries — auto-proceed (no mutation risk)
            confirm = "Y"

        if confirm == "Y":
            logger.info("HITL APPROVED — executing SQL: %s", sql)
            try:
                columns, rows = execute_sql(sql)
                print("\n  [✓] Result:")
                display_results(columns, rows)
                logger.info("SQL execution successful.")

                # ── Refresh schema store after DDL ────────────────────────
                if DDL_KEYWORDS.search(sql):
                    print("\n  [*] Refreshing schema vector store after DDL change...")
                    try:
                        refresh_vector_store(collection)
                        print("  [✓] Schema store refreshed.")
                    except Exception as ref_err:
                        print(f"  [!] Schema refresh warning: {ref_err}")
                        logger.warning("Schema refresh after DDL failed: %s", ref_err)

            except Exception as e:
                print(f"\n  [✗] Execution error: {e}")
                logger.error("SQL execution error: %s", e)
        else:
            print("\n  [!] Execution CANCELLED by user.")
            logger.info("HITL REJECTED — SQL execution cancelled by user.")

        print("\n  " + "─" * 60)


if __name__ == "__main__":
    main()

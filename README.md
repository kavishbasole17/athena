# Athena: RAG-Powered Text-to-SQL Agent

Athena is an intelligent, locally-hosted Command Line Interface (CLI) application that translates natural language questions into executable database queries. 

## 1. Is this a RAG System? (Yes!)

RAG stands for **Retrieval-Augmented Generation**. While many people associate RAG exclusively with reading PDFs or web pages, Athena is actually a textbook example of a RAG architecture applied to structured databases:

1. **Retrieval**: The system uses `SentenceTransformers` to convert the natural language question into a vector and searches `ChromaDB` for the most semantically similar data structures (the Table schemas/DDL).
2. **Augmented**: It takes the retrieved, relevant SQLite schema and injects it into the LLM's prompt context.
3. **Generation**: The `sqlcoder` language model uses that specific schema context alongside the user's question to successfully generate an accurate SQL statement.

By dynamically retrieving only the relevant tables rather than stuffing the entire database schema into the conversational context, it scales efficiently and prevents the LLM from becoming confused by irrelevant table structures. 

---

## 2. System Architecture & Workflow Design

The system follows a strict sequential pipeline divided between two primary engines: `core/db_manager.py` (The Infrastructure) and `core/llm_engine.py` (The AI Brain).

### Architectural Workflow

1. **Initialization (`setup_database` & `setup_vector_store`)**:
   - The SQLite database (`university.db`) boots and seeds standard mock data.
   - `db_manager.py` extracts the schema definition queries (DDL) of all existing tables and uses the `all-MiniLM-L6-v2` embedding model to encode them.
   - These embeddings are saved inside an offline Vector Database (`ChromaDB`).
   - `main.py` initializes local connection endpoints to the `sqlcoder` Ollama engine through LangChain (`LCEL`).

2. **The NLP Prompt Loop (Step 1)**:
   - The user inputs text: *"Fetch all entries from the table food."*
   - ChromaDB is queried against the underlying vector embeddings to pull the `food` schema context string (`retrieve_relevant_schema`).

3. **Dual Generation Algorithms (Step 2)**:
   - The retrieved schema + string enters `core/llm_engine.py`.
   - The prompt is routed identically across **Two different text-generation implementations** (Algorithm 1 and Algorithm 2) asynchronously.
   - The pipeline tests the syntactical logic of both returned output strings logically using the native SQLite `EXPLAIN {}` keyword payload, collecting performance and validation markers.

4. **Human-in-the-Loop & Execution (Step 3 & 4)**:
   - A separate plain-English Chain explains the logical outcome to the user for validation.
   - If the SQL string contains standard Read parameters (`SELECT`), the pipeline executes immediately and visualizes the row data cleanly in the UI.
   - If the SQL modifies state structurally (`ALTER`, `DROP`, `INSERT`, `UPDATE`), the HITL Guardrail pauses execution indefinitely until the User inputs `Y`. 
   - If User authorizes a structural reset (e.g. they created a new table), `refresh_vector_store` is invoked to instantly re-index ChromaDB preventing knowledge stagnation.

---

## 3. Algorithm Analysis in Detail

The system leverages two unique prompt implementations using LangChain Expression Language (LCEL) pipelines logically grouped dynamically inside `main.py`.

### Algorithm 1: The Contextual Conversational Model
This serves as the primary standard algorithm. It is structured entirely around the exact conversational fine-tuning parameters the `sqlcoder` parameter weights were explicitly trained on.
* **Prompt Engineering Strategy**: It feeds the system using specific Header blocks (`### Task`, `### Database Schema`, `### Rules`, `### Answer`) combined with a few-shot ` ```sql ` enclosure at the tail-end naturally forcing the LLM engine to logically fill-in-the-blank text completions. 
* **Strengths**: Highly dependable string consistency on complex joins or mathematically heavy queries because it taps into the strict format constraints the model evaluates best against.
* **Weaknesses**: Markdown code wrapper outputs (` ```sql SELECT... ``` `) require robust regex cleanup hooks structurally. More token-heavy due to extensive headers.

### Algorithm 2: The Constrained Semantic Filter (Alternate)
This is built as the strict secondary fallback. Instead of using the native conversational completions, this relies heavily on Semantic Zero-Shot configurations mapping explicitly toward syntax payload alone.
* **Prompt Engineering Strategy**: It feeds the raw query context but imposes a strict negative constraint logic gate within its rules (*"without markdown wrappers or explanations"*). It doesn't rely on formatting markdown completion wrappers, instructing the Model to behave strictly as an objective functional execution pipe. 
* **Strengths**: Designed to execute faster structurally because it trims conversational token outputs entirely over network requests. Evaluated natively over bare-metal SQL generation.
* **Weaknesses**: When strict contextual headers are stripped, highly fine-tuned models like `sqlcoder` occasionally crash their logic trees completely, returning empty generation streams as proven in performance tracking failure states.


## 🛠️ Technology Stack Breakdown

* **Language Model Agent**: `Ollama` hosting `sqlcoder` weights offline.
* **Orchestration**: `LangChain` to chain LLM invocations dynamically with Parsers.
* **Embedding Tooling**: `sentence-transformers` for calculating dense vector values.
* **Information Storage**: `ChromaDB` (Persistent) scaling semantic text look-ups.
* **Relational Core**: Standard `SQLite3` libraries.

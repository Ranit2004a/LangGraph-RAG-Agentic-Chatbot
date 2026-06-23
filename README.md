# LangGraph RAG Agentic Chatbot

A FastAPI-backed agentic RAG (Retrieval-Augmented Generation) chatbot powered by a LangGraph state machine. The system dynamically routes user queries, evaluates retrieved content sufficiency, queries a Pinecone vector database, and optionally fetches real-time web results via Tavily.

---

## Architecture

The chatbot operates as a hybrid search agent following a structured decision flow:

```
User Query
    │
    ▼
router_node  ──── route == end ──────────────────► END
    │
    ├── route == answer ─────────────────────────► answer_node
    │
    ├── route == rag ────────────────────────────► rag_lookup
    │                                                   │
    │                                           Judge: Sufficient?
    │                                            ├── Yes ──────────────────► answer_node
    │                                            ├── No + Web Enabled ─────► web_search ──► answer_node
    │                                            └── No + Web Disabled ────► answer_node
    │
    └── route == web ───────────────────────────► web_search ──► answer_node
```

**Key components:**
- `router_node` — classifies the query into one of four routing paths using `llama3-70b-8192` via Groq
- `rag_node` — retrieves top-5 relevant chunks from Pinecone and uses a judge LLM to evaluate sufficiency
- `web_node` — fetches real-time results via Tavily when RAG context is insufficient or not applicable
- `answer_node` — synthesizes all retrieved context into a final response
- `MemorySaver` — persists conversation history across turns within a session thread

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | FastAPI |
| **Agentic Orchestration** | LangGraph |
| **LLM Provider** | Groq (`llama3-70b-8192`) |
| **Vector Database** | Pinecone (Serverless, `us-east-1`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Web Search** | Tavily Search API |
| **Document Parsing** | PyPDFLoader |
| **Frontend** | Streamlit |
| **Python** | >= 3.13 |

---

## Project Structure

```
project/
├── pyproject.toml              # Project metadata and Python version requirements
├── requirements.txt            # All dependencies
├── .python-version             # Python environment pin
├── .env                        # API keys (not committed)
│
└── backend/
    ├── config.py               # Loads env vars (Pinecone, Groq, Tavily, embedding model)
    ├── vectorstore.py          # Embedding, indexing, and Pinecone retrieval logic
    ├── agent.py                # LangGraph state graph, nodes, routing, memory
    └── main.py                 # FastAPI app — /chat, /upload-document, /health endpoints
```

---

## Core Modules

### `vectorstore.py`

Handles document lifecycle from raw text to searchable vector chunks.

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors)
- **Pinecone index**: Auto-created as a Serverless index on `aws/us-east-1` with cosine similarity if it does not already exist
- **`get_retriever()`**: Returns a retriever object pointing to the `rag-index` index
- **`add_document(text_content)`**: Chunks input text using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) and upserts chunks to Pinecone

---

### `agent.py`

Defines the LangGraph state machine and all agent nodes.

**State schema (`AgentState`):**

| Field | Type | Description |
|---|---|---|
| `messages` | list | Full conversation history |
| `route` | string | Current routing destination (`rag`, `web`, `answer`, `end`) |
| `rag` | string | Accumulated context from Pinecone retrieval |
| `web` | string | Accumulated context from Tavily web search |
| `web_search_enabled` | bool | Runtime flag — whether web search is permitted for this request |

**Node behavior:**
- `router_node` — uses a structured output LLM call to classify the query; overrides `web` → `rag` if `web_search_enabled` is `False`
- `rag_node` — retrieves top-5 chunks; calls a separate judge LLM to evaluate whether the retrieved context is sufficient before deciding the next step
- `web_node` — invokes `web_search_tool` via Tavily; returns result summaries and source URLs
- `answer_node` — synthesizes all context (RAG + web + conversation history) into the final user-facing response

---

### `main.py`

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-document/` | Accepts a PDF file, extracts text via PyPDFLoader, indexes it to Pinecone, cleans up temp file |
| `POST` | `/chat/` | Accepts a user query and `session_id`; streams the LangGraph agent and returns the final reply + a structured execution trace (`TraceEvent`) |
| `GET` | `/health` | Standard health check |

---

## Setup

### Prerequisites

Obtain API keys for:
- [Pinecone](https://www.pinecone.io/)
- [Groq](https://console.groq.com/)
- [Tavily](https://tavily.com/)

### Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## How It Works — End to End

1. **Upload a document** via `POST /upload-document/` — the PDF is parsed, chunked, and indexed into Pinecone.
2. **Send a query** via `POST /chat/` with your `session_id` and message.
3. The `router_node` classifies the query — should it be answered directly, looked up in the knowledge base, or searched on the web?
4. If routed to RAG, the `rag_node` retrieves the top-5 relevant chunks and a judge LLM evaluates whether they are sufficient.
5. If insufficient and web search is enabled, the `web_node` fetches live results from Tavily.
6. The `answer_node` synthesizes all available context into a final response.
7. The API returns the response text alongside a full `TraceEvent` execution trace showing which nodes were visited.

---

## Roadmap

- [ ] Streamlit frontend integration
- [ ] Multi-document session management
- [ ] Support for additional file types (DOCX, TXT, Markdown)
- [ ] Streaming response tokens to the frontend in real-time
- [ ] Dockerized deployment with docker-compose

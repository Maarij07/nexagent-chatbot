# NexAgent Agentic Chatbot — Technical Explanation

## What Changed (v1 → v2)

| Aspect | v1 (RAG only) | v2 (Agentic) |
|--------|---------------|--------------|
| Architecture | Static 2-pass RAG | LangGraph ReAct agent loop |
| Canvas manipulation | ❌ None | ✅ Full CRUD via tool calls |
| Conversation memory | ❌ Stateless | ✅ Per-session MemorySaver |
| Embeddings | MD5 hash (fake) | `BAAI/bge-small-en` (real semantic) |
| Tools | None | `update_workflow_canvas`, `search_nexagent_docs`, `get_node_schema` |
| LangChain | ❌ | ✅ LangChain 0.3 |
| LangGraph | ❌ | ✅ LangGraph 0.2 (create_react_agent) |

---

## Architecture Overview

```
User Message + Canvas State
        ↓
  ┌─────────────────────────────────────────────────────┐
  │           LangGraph ReAct Agent Loop                │
  │                                                     │
  │  System Prompt (node catalog + rules embedded)      │
  │        ↓                                            │
  │  [LLM: llama-3.3-70b-versatile via Groq]           │
  │        ↓                                            │
  │  Decides which tool(s) to call:                     │
  │    ├─ update_workflow_canvas(nodes, connections)    │
  │    ├─ search_nexagent_docs(query) → Pinecone RAG   │
  │    └─ get_node_schema(node_type) → inline lookup   │
  │        ↓                                            │
  │  Tool result fed back into agent loop               │
  │        ↓                                            │
  │  Final AIMessage (no tool calls) = answer           │
  └─────────────────────────────────────────────────────┘
        ↓                    ↓
   Text answer         workflow_action
   (Markdown)          UPDATE_CANVAS payload
        ↓                    ↓
   Chat bubble         Frontend replaces canvas
```

---

## Components

### 1. LangGraph `create_react_agent`
- Standard ReAct loop: Reason → Act (call tool) → Observe → Repeat
- Terminates when the LLM produces a message with no more tool calls
- `MemorySaver` checkpointer stores message history keyed by `thread_id` (= `session_id`)
- The system prompt (containing the full node catalog + rules) is injected every turn

### 2. LangChain Tools

#### `update_workflow_canvas`
Called when the user wants to build or modify a workflow.
- Input: `nodes: List[dict]`, `connections: List[dict]`
- Validates that the first node is a trigger
- Returns a success JSON — the real payload is extracted from the tool *call arguments* (not the return value) by `_extract_workflow_action()`

#### `search_nexagent_docs`
Called for factual questions about NexAgent.
- Uses `PineconeVectorStore.similarity_search()` with `BAAI/bge-small-en` embeddings
- Returns top-4 passage chunks with source metadata
- Source names are extracted for the `sources[]` response field

#### `get_node_schema`
Called when the agent needs exact parameter names before generating JSON.
- In-memory inline schema dict for all 21 node types
- Returns schema + required/optional flags for the requested type

### 3. HuggingFace Embeddings (`BAAI/bge-small-en`)
- 384-dimensional sentence embeddings
- Replaces the old MD5 hash approach
- `normalize_embeddings=True` for cosine similarity in Pinecone
- Pre-downloaded at build time (`build.sh`) so cold starts are fast

### 4. Pinecone Vector Store
- `langchain_pinecone.PineconeVectorStore` wraps the Pinecone SDK
- Index auto-created if missing (384 dims, cosine metric, AWS us-east-1)
- `/ingest` endpoint chunks `.pdf` and `.txt` files at 800 chars and upserts with page/source metadata

### 5. Per-session Conversation Memory
- `langgraph.checkpoint.memory.MemorySaver` stores full message history in-process
- Each request passes `config={"configurable": {"thread_id": session_id}}`
- `DELETE /reset-session/{id}` clears the checkpoint for a fresh start

---

## Request → Response Flow

```
POST /query
  body: { question, session_id, current_state }
    ↓
1. Append current_state JSON to the user message text
2. Wrap in HumanMessage
3. agent.invoke(messages, config={thread_id: session_id})
   → LangGraph runs ReAct loop with MemorySaver
4. Scan result['messages'] for:
   - Last AIMessage without tool_calls → answer text
   - Any AIMessage.tool_calls targeting update_workflow_canvas → workflow_action
   - ToolMessages from search_nexagent_docs → sources
5. Return QueryResponse
```

---

## File Structure

```
nexagent-chatbot/
├── main.py                  # FastAPI app + LangGraph agent + tools
├── requirements.txt         # All Python dependencies
├── build.sh                 # Render build script (pre-caches embedding model)
├── runtime.txt              # Python 3.11
├── .env                     # API keys (never commit)
├── data/
│   ├── questions.txt        # NexAgent FAQ knowledge base
│   └── *.pdf                # Any additional PDF docs to ingest
├── index.html               # Test chat UI (handles workflow_action cards)
├── API_DOCUMENTATION.md     # Full API reference
└── TECHNICAL_EXPLANATION.md # This file
```

---

## The Three Embedding Approaches (Why We Upgraded)

| Approach | How | Quality | Speed | Production? |
|----------|-----|---------|-------|-------------|
| MD5 hash (v1) | Hash text → 384 floats | None (random-like) | Instant | ❌ |
| BAAI/bge-small-en (v2) | Transformer sentence encoder | High semantic | ~50ms | ✅ |
| OpenAI text-embedding | API call | Highest | ~200ms + cost | ✅ (paid) |

`BAAI/bge-small-en` is the right balance — fast, free, high quality, runs on CPU.

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key for LLM inference |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index (default: `nexagent-chatbot`) |
| `embedding_model_name` | HuggingFace model ID (default: `BAAI/bge-small-en`) |
| `LLM_MODEL_NAME` | Groq model name (default: `llama-3.3-70b-versatile`) |

---

## Deployment on Render

1. Push code to GitHub
2. Render detects `requirements.txt` + `runtime.txt`
3. `build.sh` runs: installs deps + pre-caches embedding model
4. Server starts: `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Call `POST /ingest` once to populate Pinecone with your docs

> **Note:** `sentence-transformers` pulls PyTorch (~1 GB). Use Render's **Standard** plan or higher for the build step.

---

## Key Design Decisions

- **Why LangGraph over plain LangChain?** LangGraph's stateful graph gives us automatic retry loops, clean tool-call handling, and built-in memory checkpointing without manual state management.
- **Why inject `current_state` in the message?** The canvas changes every turn (the user might add nodes manually). Injecting it fresh each turn is more reliable than trusting stale history alone.
- **Why extract `workflow_action` from tool_calls (not tool return)?** The LLM's tool_call arguments contain the exact nodes/connections it decided on — this is the ground truth, not the validation return value.
- **Why `BAAI/bge-small-en`?** 384 dims matches the existing Pinecone index, runs fast on CPU, and gives real semantic search (unlike the old hash approach).

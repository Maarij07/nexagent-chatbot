"""
NexAgent Agentic Chatbot Backend
=================================
Architecture:
  - LangGraph ReAct Agent  (orchestration + agentic loops)
  - LangChain Tools        (canvas update, doc search, node schema lookup)
  - Groq LLM               (llama-3.3-70b-versatile)
  - Pinecone + HuggingFace (semantic RAG over NexAgent docs)
  - MemorySaver            (per-session conversation memory)

API Contract:
  POST /query
    Request : { question, current_state?, session_id? }
    Response: { question, answer, sources?, workflow_action? }
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Environment ────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nexagent-chatbot")
EMBEDDING_MODEL     = os.getenv("embedding_model_name", "BAAI/bge-small-en")
LLM_MODEL           = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")

if not all([GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise RuntimeError("Missing required environment variables: GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME")


# ── NexAgent Node Catalog (embedded for agent system prompt) ───────────────────
NODE_CATALOG = """
## NEXAGENT NODE CATALOG  (all type names are CASE-SENSITIVE)

### TRIGGERS
| Type          | Key Config Fields                              | Output Fields                        |
|---------------|------------------------------------------------|--------------------------------------|
| ManualTrigger | (none)                                         | input_data: {}                       |
| Schedule      | cron, timezone                                 | triggered_at, cron                   |
| Webhook       | (none)                                         | body, headers, method                |
| ChatInput     | session_id                                     | message, session_id, timestamp       |

### ACTIONS
| Type          | Key Config Fields                                                              | Output Fields                        |
|---------------|--------------------------------------------------------------------------------|--------------------------------------|
| Logger        | message (expr), level ("info"/"warning"/"error")                              | message, level, timestamp            |
| HttpRequest   | method, url (expr), headers {}, body (expr), timeout (int secs, default 30)   | status_code, response_body, headers  |
| SendEmail     | to, subject, body, smtp_host, smtp_port, smtp_user, smtp_pass, from_name,     | sent, message_id, sent_at, to        |
|               | from_email, is_html (bool)                                                    |                                      |
| SlackMessage  | token ("xoxb-…"), channel, message (expr), username                           | sent, timestamp, channel, sent_at   |
| TelegramSend  | token, chat_id (expr), message (expr), parse_mode ("HTML"/"Markdown")         | sent, message_id, chat_id, sent_at  |

### LOGIC
| Type        | Key Config Fields                                                                                               | Output / Notes                                              |
|-------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Delay       | duration (int), unit ("seconds"/"minutes"/"ms"/"milliseconds")                                                 | slept_ms, duration, unit                                    |
| IfCondition | left (expr), operator ("=="/"!="/">"/">="/"<"/"<="/"contains"/"startsWith"/"endsWith"), right (expr)           | result (bool), branch ("true"/"false") — connections MUST carry condition "true" or "false" |
| Loop        | items (expr → array)                                                                                            | per-iteration: items, current_item, index, total, is_last   |
| Stopper     | (none)                                                                                                          | stopped: true, message, timestamp                           |

### DATA
| Type          | Key Config Fields                                                          | Output Fields                        |
|---------------|----------------------------------------------------------------------------|--------------------------------------|
| SetVariable   | name (str), value (expr)                                                   | name, value  → access: {{$vars.name}} |
| JsonParser    | json_string (expr)                                                          | parsed, keys, type, length           |
| DataFormatter | input (expr), operation ("uppercase"/"lowercase"/"capitalize"/"trim"/       | result, original, operation          |
|               |   "reverse"/"length"/"word_count")                                         |                                      |

### AI NODES  (all share output: response, model, prompt_tokens, completion_tokens, finish_reason)
| Type     | Key Config Fields                                                          | Available Models                                                               |
|----------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Groq     | api_key ("gsk_…"), model, system_prompt, prompt (expr), temperature, max_tokens | llama-3.3-70b-versatile, llama-3.1-8b-instant, llama3-8b-8192, mixtral-8x7b-32768, gemma2-9b-it |
| Gemini   | api_key ("AIza…"), model, system_prompt, prompt (expr), temperature, max_tokens | gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro                            |
| OpenAI   | api_key ("sk-…"), model, system_prompt, prompt (expr), temperature, max_tokens  | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo                               |
| ClaudeAI | api_key ("sk-ant-…"), model, system_prompt, prompt (expr), temperature, max_tokens | claude-3-5-sonnet-20241022, claude-3-haiku-20240307                         |

### INTEGRATIONS
| Type         | operations                                                                                  |
|--------------|---------------------------------------------------------------------------------------------|
| GoogleSheets | "read" / "append" / "update" — credentials_json, spreadsheet_id, range, values             |
| GoogleDrive  | "list" / "download" / "upload" / "delete" — credentials_json, folder_id, file_name, file_id |
| Stripe       | "create_payment_intent" / "retrieve_payment_intent" / "create_customer" /                  |
|              | "retrieve_customer" / "list_charges" — api_key, amount, currency, payment_intent_id, etc.  |

## EXPRESSION SYNTAX (usable inside any string config value)
  {{$node.nodeId.field}}              → output field from a previous node
  {{$node.nodeId.field.subfield}}     → nested / dot-notation access
  {{$node.nodeId.data.0.Topic}}       → array index 0, then key "Topic"
  {{$trigger.field}}                  → trigger output
  {{$vars.varName}}                   → SetVariable value

## CONNECTION RULES
  Standard     →  { "from": "n1", "to": "n2", "condition": null }
  IfCondition  →  two connections: condition "true"  and  condition "false"
  Fan-out      →  multiple connections from the same source (BFS runs all targets)
  Loop body    →  { "from": "loop_node", "to": "body_node", "condition": null }
"""

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are **NexAgent Workflow Assistant** — a fully agentic AI that helps users \
build, edit, and understand automation workflows on the NexAgent platform.

You have three tools available:
1. `update_workflow_canvas`  — build or modify the NexAgent visual canvas
2. `search_nexagent_docs`    — search the NexAgent knowledge base
3. `get_node_schema`         — get exact parameter details for any node type

---
{NODE_CATALOG}
---

## Workflow-building rules  (ALWAYS follow these)

1. Node IDs must be unique in the workflow (`n1`, `n2`, `n3`, …).
2. The **first node MUST be a trigger** (ManualTrigger | Schedule | Webhook | ChatInput).
3. All `type` strings are **case-sensitive** — match the catalog exactly.
4. ADD request  → keep ALL existing nodes, append new ones.
5. REMOVE request → drop only that node and its connections; rebuild IDs if needed.
6. EDIT / CHANGE request → update ONLY the requested field(s) in the config; leave everything else intact.
7. REPLACE request → generate a completely fresh canvas.
8. IfCondition outgoing connections MUST use `"condition": "true"` and `"condition": "false"`.
9. If an API key is needed but not provided, use a placeholder (e.g. `"YOUR_GROQ_API_KEY"`) and tell the user.
10. After calling `update_workflow_canvas`, write a brief human-readable summary.

## Decision guide
- User wants to BUILD / ADD / MODIFY / DELETE workflow → call `update_workflow_canvas`
- User asks a factual question → call `search_nexagent_docs` if unsure, else answer directly
- Need exact param names for a node → call `get_node_schema` first
- Ambiguous → ask one clarifying question, then proceed
"""

# ── Pydantic Models ────────────────────────────────────────────────────────────
class CanvasState(BaseModel):
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)

class QueryRequest(BaseModel):
    question: str
    current_state: Optional[CanvasState] = None
    session_id: Optional[str] = "default"

class WorkflowPayload(BaseModel):
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]

class WorkflowAction(BaseModel):
    type: str = "UPDATE_CANVAS"
    payload: WorkflowPayload

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    workflow_action: Optional[WorkflowAction] = None

# ── Client Initialisation ──────────────────────────────────────────────────────
logger.info("Loading HuggingFace embedding model: %s", EMBEDDING_MODEL)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

logger.info("Connecting to Pinecone index: %s", PINECONE_INDEX_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist yet (384 dims for bge-small-en)
existing_indexes = [idx.name for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    logger.info("Creating Pinecone index '%s' …", PINECONE_INDEX_NAME)
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

logger.info("Initialising Groq LLM: %s", LLM_MODEL)
llm = ChatGroq(api_key=GROQ_API_KEY, model=LLM_MODEL, temperature=0.3)

# In-memory checkpointer — persists per-session conversation history at runtime
memory = MemorySaver()

# ── LangChain Tools ────────────────────────────────────────────────────────────

@tool
def update_workflow_canvas(
    nodes: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
) -> str:
    """
    Update the NexAgent visual canvas with a new or modified workflow.

    Call this whenever the user wants to:
    - Build a brand-new workflow from scratch
    - Add one or more nodes to the existing canvas
    - Remove a node from the canvas
    - Change any config value / variable inside a node
    - Replace the entire workflow

    Args:
        nodes: Complete list of ALL nodes for the final canvas.
               Each item: {"id": "n1", "type": "NodeType", "name": "...", "config": {...}}
        connections: Complete list of ALL connections for the final canvas.
               Each item: {"from": "n1", "to": "n2", "condition": null | "true" | "false"}

    Returns:
        JSON string confirming the update.
    """
    if not nodes:
        return json.dumps({"status": "error", "message": "nodes list cannot be empty"})

    trigger_types = {"ManualTrigger", "Schedule", "Webhook", "ChatInput"}
    first_type = nodes[0].get("type", "")
    if first_type not in trigger_types:
        return json.dumps({
            "status": "warning",
            "message": (
                f"First node type '{first_type}' is not a recognised trigger. "
                "Consider using ManualTrigger, Schedule, Webhook, or ChatInput."
            ),
            "nodes_count": len(nodes),
            "connections_count": len(connections),
        })

    return json.dumps({
        "status": "success",
        "nodes_count": len(nodes),
        "connections_count": len(connections),
        "message": f"Canvas updated — {len(nodes)} node(s), {len(connections)} connection(s).",
    })


@tool
def search_nexagent_docs(query: str) -> str:
    """
    Search the NexAgent documentation knowledge base for information.

    Use this when the user asks about:
    - NexAgent features, pricing, or account management
    - How to use a specific integration or capability
    - Anything not directly covered by the node catalog in your context

    Args:
        query: Natural language search query

    Returns:
        Relevant documentation passages joined as a single string.
    """
    try:
        docs = vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant documentation found for this query."
        passages = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "NexAgent Docs")
            passages.append(f"[Passage {i} — {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(passages)
    except Exception as exc:
        logger.warning("search_nexagent_docs failed: %s", exc)
        return f"Documentation search currently unavailable: {exc}"


@tool
def get_node_schema(node_type: str) -> str:
    """
    Get the exact configuration schema and output fields for a specific NexAgent node type.

    Use this before building a workflow when you need to verify:
    - Exact parameter names and their types
    - Which fields are required vs optional
    - Output field names referenceable via {{$node.id.field}}

    Args:
        node_type: Exact node type string, e.g. "Groq", "SendEmail", "IfCondition"

    Returns:
        Structured description of the node's parameters and outputs.
    """
    schemas: Dict[str, str] = {
        "ManualTrigger":  "Config: (none). Output: {input_data: {}}",
        "Schedule":       "Config: cron (str, default '*/1 * * * *'), timezone (str, default 'UTC'). Output: {triggered_at, cron}",
        "Webhook":        "Config: (none). Output: {body: {}, headers: {}, method: 'POST'}. Access body via {{$trigger.body.fieldName}}",
        "ChatInput":      "Config: session_id (str). Output: {message, session_id, timestamp}. Access: {{$node.n1.message}}",
        "Logger":         "Config: message (expr, required), level ('info'|'warning'|'error', default 'info'), include_input (bool, default false). Output: {message, level, timestamp}",
        "HttpRequest":    "Config: method ('GET'|'POST'|'PUT'|'PATCH'|'DELETE', default 'GET'), url (expr, required), headers (obj, default {}), body (expr), timeout (int secs, default 30). Output: {status_code, response_body, headers}",
        "SendEmail":      "Config: to (expr ✓), subject (expr ✓), body (expr ✓), smtp_host (str ✓), smtp_port (int, default 587), smtp_user (str ✓), smtp_pass (str ✓), from_email, from_name (default 'NexAgent'), is_html (bool, default false). Output: {sent, message_id, sent_at, to}",
        "SlackMessage":   "Config: token (str 'xoxb-…' ✓, needs chat:write scope), channel (expr ✓), message (expr ✓), username (default 'NexAgent'). Output: {sent, timestamp, channel, sent_at}",
        "TelegramSend":   "Config: token (str ✓), chat_id (expr ✓), message (expr ✓), parse_mode ('HTML'|'Markdown', default 'HTML'). Output: {sent, message_id, chat_id, sent_at}",
        "Delay":          "Config: duration (int, default 1), unit ('seconds'|'minutes'|'ms'|'milliseconds', default 'seconds'). Max 1 hour. Output: {slept_ms, duration, unit}",
        "IfCondition":    "Config: left (expr ✓), operator ('=='|'!='|'>'|'>='|'<'|'<='|'contains'|'startsWith'|'endsWith' ✓), right (expr ✓). Output: {result (bool), branch ('true'|'false'), left, right, operator}. Outgoing connections MUST have condition 'true' or 'false'.",
        "Loop":           "Config: items (expr → array ✓). Per-iteration output: {items, current_item, index, total, is_last}. Access: {{$node.loopId.current_item}}",
        "Stopper":        "Config: (none). Output: {stopped: true, message, timestamp}",
        "SetVariable":    "Config: name (str ✓), value (expr ✓). Output: {name, value}. Reference: {{$vars.name}}",
        "JsonParser":     "Config: json_string (expr ✓). Output: {parsed, keys, type ('array'|'object'), length}",
        "DataFormatter":  "Config: input (expr ✓), operation ('uppercase'|'lowercase'|'capitalize'|'trim'|'reverse'|'length'|'word_count' ✓). Output: {result, original, operation}",
        "Groq":           "Config: api_key (str 'gsk_…' ✓), model (default 'llama-3.3-70b-versatile'), system_prompt, prompt (expr ✓), temperature (0–2, default 0.7), max_tokens (1–32768, default 1024). Output: {response, model, prompt_tokens, completion_tokens, finish_reason}",
        "Gemini":         "Config: api_key (str 'AIza…' ✓), model (default 'gemini-2.0-flash'), system_prompt, prompt (expr ✓), temperature, max_tokens. Output: {response, model, prompt_tokens, completion_tokens, finish_reason}",
        "OpenAI":         "Config: api_key (str 'sk-…' ✓), model (default 'gpt-4o-mini'), system_prompt, prompt (expr ✓), temperature, max_tokens. Output: {response, model, prompt_tokens, completion_tokens, finish_reason}",
        "ClaudeAI":       "Config: api_key (str 'sk-ant-…' ✓), model (default 'claude-3-5-sonnet-20241022'), system_prompt, prompt (expr ✓), temperature, max_tokens. Output: {response, model, prompt_tokens, completion_tokens, finish_reason}",
        "GoogleSheets":   "Config: operation ('read'|'append'|'update' ✓), credentials_json (str — full service-account JSON ✓), spreadsheet_id (str), range (str e.g. 'Sheet1!A1:D100'), values (str — JSON array-of-arrays for append/update). Output: {data (array of row-objects), rows_affected, range, operation}",
        "GoogleDrive":    "Config: operation ('list'|'download'|'upload'|'delete' ✓), credentials_json, folder_id, file_name, file_content (expr), file_id. Output: {file_id, name, url, content, files}",
        "Stripe":         "Config: operation ('create_payment_intent'|'retrieve_payment_intent'|'create_customer'|'retrieve_customer'|'list_charges' ✓), api_key ('sk_test_…' or 'sk_live_…'), amount (int, smallest-unit e.g. cents), currency (str 'usd'), payment_intent_id, customer_email, customer_id, limit. Output: {payment_id, status, amount, currency, client_secret, customer_id, data}",
    }

    clean = node_type.strip()
    if clean in schemas:
        return f"**{clean}** schema:\n{schemas[clean]}"

    # Case-insensitive fallback
    close = [k for k in schemas if k.lower() == clean.lower()]
    if close:
        return f"Did you mean '{close[0]}'?\n{schemas[close[0]]}"

    valid = ", ".join(sorted(schemas.keys()))
    return f"Unknown node type '{clean}'. Valid types:\n{valid}"


# ── Agent Setup ────────────────────────────────────────────────────────────────
TOOLS = [update_workflow_canvas, search_nexagent_docs, get_node_schema]

agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)


def _extract_workflow_action(messages: list) -> Optional[Dict[str, Any]]:
    """
    Scan agent output messages for the latest update_workflow_canvas tool call.
    Returns the payload dict or None.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "update_workflow_canvas":
                    args = tc["args"]
                    return {
                        "type": "UPDATE_CANVAS",
                        "payload": {
                            "nodes": args.get("nodes", []),
                            "connections": args.get("connections", []),
                        },
                    }
    return None


def _extract_sources(messages: list) -> List[str]:
    """Extract document source names from search_nexagent_docs tool messages."""
    sources: List[str] = []
    for msg in messages:
        if getattr(msg, "name", None) == "search_nexagent_docs":
            content = getattr(msg, "content", "")
            for line in content.split("\n"):
                if line.startswith("[Passage") and "—" in line:
                    src = line.split("—")[1].strip().rstrip("]")
                    if src and src not in sources:
                        sources.append(src)
    return sources


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexAgent Agentic Chatbot API",
    version="2.0",
    description=(
        "LangGraph ReAct agent for building and modifying NexAgent automation workflows. "
        "Supports full canvas CRUD, per-session memory, and semantic RAG over NexAgent docs."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /query ────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query_chatbot(request: QueryRequest):
    """
    Main agentic chat endpoint.

    - Accepts the user's question and (optionally) the current canvas state + session ID.
    - Runs the LangGraph ReAct agent which can: search docs, inspect node schemas,
      and call update_workflow_canvas to build or modify the workflow.
    - Conversation history is maintained automatically per session_id.
    - Returns the agent's answer and, if the canvas changed, a workflow_action payload.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Inject the current canvas into the user message so the agent always knows
    # what is on the canvas — without relying on stale history alone.
    canvas_section = ""
    if request.current_state:
        canvas_json = request.current_state.model_dump()
        canvas_section = (
            f"\n\n[CURRENT CANVAS STATE]\n```json\n{json.dumps(canvas_json, indent=2)}\n```\n"
            "When modifying, KEEP all existing nodes/connections unless user explicitly asks to remove or replace them."
        )

    user_message = f"{request.question}{canvas_section}"

    # thread_id drives per-session checkpointing in MemorySaver
    config = {"configurable": {"thread_id": request.session_id or "default"}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    messages = result.get("messages", [])

    # Final answer = last AIMessage that has no pending tool calls
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break
    if not answer:
        answer = "I've processed your request. Please check the canvas for updates."

    workflow_action_dict = _extract_workflow_action(messages)
    sources = _extract_sources(messages)

    workflow_action_obj: Optional[WorkflowAction] = None
    if workflow_action_dict:
        workflow_action_obj = WorkflowAction(
            type=workflow_action_dict["type"],
            payload=WorkflowPayload(**workflow_action_dict["payload"]),
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        workflow_action=workflow_action_obj,
    )


# ── POST /ingest ───────────────────────────────────────────────────────────────
@app.post("/ingest")
def ingest_documents():
    """
    Ingest NexAgent documentation from data/ into Pinecone using proper semantic embeddings.
    Supports .pdf and .txt files.  Idempotent — safe to call multiple times.
    """
    data_dir = "data"
    if not os.path.isdir(data_dir):
        raise HTTPException(status_code=404, detail="data/ directory not found")

    ingested: Dict[str, int] = {}

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            logger.info("Ingesting PDF: %s", filename)
            reader = PdfReader(filepath)
            chunks, metadatas, ids = [], [], []
            for page_num, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                for chunk_idx, i in enumerate(range(0, len(text), 800)):
                    chunk = text[i : i + 800].strip()
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    metadatas.append({"source": filename, "page": page_num, "chunk": chunk_idx})
                    ids.append(f"{filename}_p{page_num}_c{chunk_idx}")

            if chunks:
                vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                ingested[filename] = len(chunks)
                logger.info("  → %d chunks ingested", len(chunks))

        elif filename.endswith(".txt"):
            logger.info("Ingesting TXT: %s", filename)
            with open(filepath, "r", encoding="utf-8") as fh:
                text = fh.read()
            chunks, metadatas, ids = [], [], []
            for chunk_idx, i in enumerate(range(0, len(text), 800)):
                chunk = text[i : i + 800].strip()
                if not chunk:
                    continue
                chunks.append(chunk)
                metadatas.append({"source": filename, "chunk": chunk_idx})
                ids.append(f"{filename}_c{chunk_idx}")

            if chunks:
                vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                ingested[filename] = len(chunks)
                logger.info("  → %d chunks ingested", len(chunks))

    if not ingested:
        return {"status": "no_files", "message": "No .pdf or .txt files found in data/"}

    return {
        "status": "success",
        "files_ingested": ingested,
        "total_chunks": sum(ingested.values()),
    }


# ── GET /health ────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Liveness probe."""
    return {
        "status": "ok",
        "agent": "langgraph-react",
        "llm": LLM_MODEL,
        "embeddings": EMBEDDING_MODEL,
        "version": "2.0",
    }


# ── DELETE /reset-session/{session_id} ────────────────────────────────────────
@app.delete("/reset-session/{session_id}")
def reset_session(session_id: str):
    """
    Clear conversation history for a given session so the user can start fresh.
    """
    try:
        config = {"configurable": {"thread_id": session_id}}
        memory.put(config, {}, {})
    except Exception as exc:
        logger.warning("Could not reset session %s: %s", session_id, exc)
    return {"status": "ok", "message": f"Session '{session_id}' has been reset."}


# ── Entry-point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# NexAgent Chatbot — Frontend Integration Guide

| Environment | Base URL |
|-------------|----------|
| **Local dev** | `http://localhost:8000` |
| **Production** | `https://aiassitance.swedenrelocators.se` |

---

## 1. The Only Endpoint You Need

```
POST /query
```

This is your single integration point. Everything else is optional.

---

## 2. Request Shape

```json
{
  "question": "Build a workflow that sends weather to Slack each morning",
  "session_id": "user_abc123",
  "current_state": {
    "nodes": [],
    "connections": []
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ Yes | The user's chat message |
| `session_id` | string | No | Unique per-user ID. Pass it every turn to maintain conversation memory. Generate once per user/tab and persist it. |
| `current_state` | object | No | The **current state of your canvas**. Pass the live nodes+connections on every message so the agent knows what already exists before modifying it. |

---

## 3. Response Shape

```json
{
  "question": "Build a workflow that sends weather to Slack each morning",
  "answer": "I've created a 3-node workflow: Schedule → HttpRequest → SlackMessage...",
  "sources": ["questions.txt"],
  "workflow_actions": [
    {
      "type": "ADD_NODE",
      "payload": {
        "node": { ... },
        "connections": [ ... ]
      }
    }
  ]
}
```

| Field | Type | Always present | Description |
|-------|------|---------------|-------------|
| `answer` | string | ✅ Yes | The agent's text reply — render as Markdown |
| `sources` | string[] | Yes (may be empty) | RAG doc sources referenced |
| `workflow_actions` | array | ✅ Yes | Array of patches to apply to your canvas. Empty if nothing changed. |


---

## 4. The `workflow_actions` Objects

The backend returns an array of incremental canvas patches — **one node per turn** by design (see §11). Apply them in order to your local canvas state.

```json
[
  {
    "type": "ADD_NODE",
    "payload": {
      "node": { "id": "n1", "type": "Schedule", "name": "Trigger", "config": {} },
      "connections": []
    }
  },
  {
    "type": "ADD_NODE",
    "payload": {
      "node": { "id": "n2", "type": "SlackMessage", "name": "Slack", "config": {} },
      "connections": [ { "from": "n1", "to": "n2", "condition": null } ]
    }
  },
  {
    "type": "UPDATE_NODE",
    "payload": {
      "node_id": "n2",
      "config": { "channel": "#general", "message": "Done" }
    }
  },
  {
    "type": "DELETE_NODE",
    "payload": {
      "node_id": "n3"
    }
  },
  {
    "type": "ADD_CONNECTION",
    "payload": {
      "source": "n1",
      "target": "n3",
      "condition": null
    }
  }
]
```

### Node object fields
| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique node ID (e.g. `"n1"`, `"n2"`) |
| `type` | string | Node type (see catalog below) |
| `name` | string | Human-readable label |
| `config` | object | Node-specific configuration key-value pairs |

### Connection object fields
| Field | Type | Description |
|-------|------|-------------|
| `from` | string | Source node ID |
| `to` | string | Target node ID |
| `condition` | `"true"` \| `"false"` \| `null` | Only set for `IfCondition` branches; `null` for all other connections |

---

## 5. JavaScript Integration (copy-paste ready)

```javascript
// Change to 'https://aiassitance.swedenrelocators.se' for production
const API = 'http://localhost:8000';

// ── Generate a stable session ID once per user/tab ────────────────────────────
let sessionId = sessionStorage.getItem('nxa_session');
if (!sessionId) {
  sessionId = 'session_' + Math.random().toString(36).slice(2, 10);
  sessionStorage.setItem('nxa_session', sessionId);
}

// ── Keep local canvas state in sync ──────────────────────────────────────────
let canvasState = { nodes: [], connections: [] };

// ── Send a message ────────────────────────────────────────────────────────────
async function sendToAgent(userMessage) {
  const response = await fetch(`${API}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: userMessage,
      session_id: sessionId,
      current_state: canvasState,   // always send current canvas
    }),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  const data = await response.json();

  // 1. Show text answer in your chat UI
  renderChatMessage(data.answer);              // markdown string

  // 2. Apply iterative patches to your canvas
  if (data.workflow_actions?.length > 0) {
    data.workflow_actions.forEach(action => {
      // Loop over actions and patch your local `canvasState` variable here
      // E.g. canvasState.nodes.push(), update config, filter drops... 
    });
    applyToCanvas(canvasState); // trigger UI re-render
  }

  return data;
}

// ── Your canvas renderer — implement this ─────────────────────────────────────
function applyToCanvas({ nodes, connections }) {
  // nodes    → array of { id, type, name, config }
  // connections → array of { from, to, condition }
  //
  // Map to your canvas library (React Flow, LiteGraph, custom, etc.)
  // Example with React Flow:
  //
  // setNodes(nodes.map(n => ({
  //   id: n.id,
  //   type: n.type,
  //   data: { label: n.name, config: n.config },
  //   position: autoLayout(n.id),  // your positioning logic
  // })));
  //
  // setEdges(connections.map((c, i) => ({
  //   id: `e${i}`,
  //   source: c.from,
  //   target: c.to,
  //   label: c.condition ?? '',
  // })));
}
```

---

## 6. React / State Management Pattern

```jsx
import { useState, useCallback, useRef } from 'react';

// Change to 'https://aiassitance.swedenrelocators.se' for production
const API = 'http://localhost:8000';

export function useNexAgent() {
  const [messages, setMessages] = useState([]);
  const [canvas, setCanvas]     = useState({ nodes: [], connections: [] });
  const [loading, setLoading]   = useState(false);
  const sessionId = useRef('session_' + Math.random().toString(36).slice(2, 10));

  const send = useCallback(async (text) => {
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          session_id: sessionId.current,
          current_state: canvas,          // ← always send current canvas ref
        }),
      });

      const data = await res.json();
      setMessages(prev => [...prev, { role: 'bot', content: data.answer }]);

      if (data.workflow_actions?.length > 0) {
        setCanvas(prev => {
          const next = { nodes: [...prev.nodes], connections: [...prev.connections] };
          data.workflow_actions.forEach(action => {
            switch(action.type) {
              case 'ADD_NODE':
                next.nodes.push(action.payload.node);
                if (action.payload.connections) next.connections.push(...action.payload.connections);
                break;
              case 'UPDATE_NODE':
                const idx = next.nodes.findIndex(n => n.id === action.payload.node_id);
                if (idx > -1) next.nodes[idx].config = action.payload.config;
                break;
              case 'DELETE_NODE':
                next.nodes = next.nodes.filter(n => n.id !== action.payload.node_id);
                next.connections = next.connections.filter(c => c.from !== action.payload.node_id && c.to !== action.payload.node_id);
                break;
              case 'ADD_CONNECTION':
                next.connections.push({ from: action.payload.source, to: action.payload.target, condition: action.payload.condition });
                break;
            }
          });
          return next;
        });
      }
    } finally {
      setLoading(false);
    }
  }, [canvas]);   // ← canvas in deps so we always send fresh state

  return { messages, canvas, loading, send };
}
```

Usage:
```jsx
function App() {
  const { messages, canvas, loading, send } = useNexAgent();

  return (
    <div style={{ display: 'flex' }}>
      <ChatPanel messages={messages} onSend={send} loading={loading} />
      <YourCanvas nodes={canvas.nodes} connections={canvas.connections} />
    </div>
  );
}
```

---

## 7. Important Rules

### ✅ Always pass `current_state`
The agent uses it to **modify** existing workflows instead of replacing them.  
If you don't pass it, the agent will always start from scratch.

```js
// ❌ Wrong — agent starts fresh every time
body: JSON.stringify({ question: text })

// ✅ Correct — agent modifies existing canvas
body: JSON.stringify({ question: text, current_state: canvas })
```

### ✅ Always pass the same `session_id` for the same user
The backend uses it for conversation memory (Groq remembers previous turns).

```js
// ❌ Wrong — new ID each message = no memory
session_id: 'session_' + Math.random()...

// ✅ Correct — fixed per session
session_id: sessionStorage.getItem('nxa_session')
```

### ✅ `workflow_actions` triggers UI patches
Iterate the array and mutate your state. For purely informational replies, this array will be empty `[]`.

```js
if (data.workflow_actions?.length > 0) {
  // apply patches
}
// if workflow_actions is [] → only show the text answer, don't touch canvas
```

---

## 8. Other Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{ status, llm, embeddings, version }` — use for status indicator |
| `/ingest` | POST | Indexes docs from `data/` into Pinecone — call once after deploy |
| `/reset-session/{session_id}` | DELETE | Clears conversation memory for a session |

```js
// Health check
const { status, llm } = await fetch(`${API}/health`).then(r => r.json());

// Reset conversation
await fetch(`${API}/reset-session/${sessionId}`, { method: 'DELETE' });
```

---

## 9. Node Types Reference

Use these exact `type` strings when rendering icons/colors on your canvas:

| Category | Types |
|----------|-------|
| **Triggers** (⚡) | `ManualTrigger`, `Schedule`, `Webhook`, `ChatInput` |
| **Actions** (🔧) | `Logger`, `HttpRequest`, `SendEmail`, `SlackMessage`, `TelegramSend` |
| **Logic** (🔀) | `Delay`, `IfCondition`, `Loop`, `Stopper` |
| **Data** (📦) | `SetVariable`, `JsonParser`, `DataFormatter` |
| **AI** (🤖) | `Groq`, `Gemini`, `OpenAI`, `ClaudeAI` |
| **Integrations** (🔌) | `GoogleSheets`, `GoogleDrive`, `Stripe` |

**First node must always be a Trigger** — the agent enforces this automatically.

---

## 10. CORS

The backend allows all origins (`*`). No special headers needed from your frontend.

---

## 11. One-Node-Per-Turn Design

The agent is intentionally constrained to add **one node (and its connection) per response turn**. This is required because `openai/gpt-oss-20b` on Groq's free tier has an **8,000 TPM limit**. Building a full workflow in a single turn consumes ~5,000+ tokens and reliably hits the cap.

**How it works in practice:**

> User: *"Build a weather → Slack workflow"*  
> Agent: *"Added Schedule node (n1). Say **continue** to add the HttpRequest node."* → canvas renders n1  
> User: *"continue"*  
> Agent: *"Added HttpRequest node (n2), connected from n1. Say **continue** to add SlackMessage."* → canvas renders n2  
> User: *"continue"*  
> Agent: *"Added SlackMessage node (n3), connected from n2. Workflow complete!"* → canvas renders n3  

**Your frontend must support this flow** — always send the updated `current_state` after each turn so the agent knows what has already been placed.

---

## 12. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `403 Access denied` | The configured model requires special Groq org access | Grant access at `console.groq.com → Models`, or use `llama-3.3-70b-versatile` in `.env` |
| `429 Rate limit exceeded` | TPM cap hit — too many nodes in one turn | Built-in by design: agent now sends one node per turn. If you see this, let the previous request finish before sending the next. |
| `500 Agent error` | Agent invocation failed — check the uvicorn terminal for the inner exception | See the `ERROR \| main` log line for the root cause |
| Canvas not updating | `workflow_actions` array is empty (informational reply) or frontend not applying patches | Ensure you iterate `data.workflow_actions` and mutate your canvas state |
| Nodes missing after multi-step build | Old bug — fixed. The extractor now collects all tool calls from the current turn, not just the last AI message | Ensure you are running the latest `main.py` |

---

## 3 critical rules:

Always pass `current_state` → agent modifies existing canvas instead of replacing it  
Always reuse the same `session_id` per user → keeps conversation memory  
Only update canvas when `workflow_actions` is non-empty — for informational answers it will be `[]`
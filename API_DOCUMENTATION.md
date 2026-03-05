# NexAgent Agentic Chatbot API Documentation

> **Version 2.0** — LangGraph ReAct Agent + Groq + Pinecone  
> Architecture upgrade: from basic RAG → fully agentic with canvas manipulation, per-session memory, and semantic search.

## Base URL
```
https://nexagent-chatbot.onrender.com
```
*(or `http://127.0.0.1:8000` for local development)*

---

## Endpoints

### 1. Query — Main Agentic Endpoint
**Method:** `POST`  
**Path:** `/query`

The agent can:
- Answer questions about NexAgent using the knowledge base (RAG)
- Build brand-new workflows from plain-English descriptions
- Add / remove / edit nodes and connections on the existing canvas
- Change any variable or config field inside a specific node
- Maintain multi-turn conversation history per `session_id`

#### Request Body
```json
{
  "question": "Build a workflow that fetches weather and sends it to Slack every morning",
  "session_id": "user_abc123",
  "current_state": {
    "nodes": [
      { "id": "n1", "type": "ManualTrigger", "config": {} }
    ],
    "connections": []
  }
}
```

| Field           | Type     | Required | Description                                                                          |
|-----------------|----------|----------|--------------------------------------------------------------------------------------|
| `question`      | string   | ✅       | The user's message or instruction                                                    |
| `session_id`    | string   | ❌       | Stable ID for this user/tab — enables conversation memory. Defaults to `"default"` |
| `current_state` | object   | ❌       | The current canvas (`nodes` + `connections`). Agent uses this context when editing. |

#### Response Body
```json
{
  "question": "Build a workflow that fetches weather and sends it to Slack every morning",
  "answer": "I've built a 3-node workflow for you! A Schedule trigger fires at 8 AM daily, fetches the weather via HttpRequest, and posts it to your Slack channel.",
  "sources": [],
  "workflow_action": {
    "type": "UPDATE_CANVAS",
    "payload": {
      "nodes": [
        { "id": "n1", "type": "Schedule", "config": { "cron": "0 8 * * *", "timezone": "UTC" } },
        { "id": "n2", "type": "HttpRequest", "config": { "method": "GET", "url": "https://wttr.in/London?format=3" } },
        { "id": "n3", "type": "SlackMessage", "config": { "token": "YOUR_SLACK_TOKEN", "channel": "#general", "message": "🌤️ Weather: {{$node.n2.response_body}}" } }
      ],
      "connections": [
        { "from": "n1", "to": "n2", "condition": null },
        { "from": "n2", "to": "n3", "condition": null }
      ]
    }
  }
}
```

| Field             | Type            | Description                                                                           |
|-------------------|-----------------|---------------------------------------------------------------------------------------|
| `question`        | string          | Echoed back from the request                                                          |
| `answer`          | string          | Human-readable response from the agent (Markdown)                                    |
| `sources`         | string[]        | Document sources used if RAG was invoked (empty if no search was done)               |
| `workflow_action` | object \| null  | Present when the agent updated the canvas. Frontend should replace canvas with payload |

> **Frontend integration:** If `workflow_action` is present, wipe the canvas and replace it with `payload.nodes` and `payload.connections`.

#### JavaScript Example
```javascript
const sessionId = 'user_' + Math.random().toString(36).slice(2, 10);
let currentCanvas = { nodes: [], connections: [] };

async function sendMessage(question) {
  const res = await fetch('https://nexagent-chatbot.onrender.com/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      current_state: currentCanvas,
    }),
  });

  const data = await res.json();
  console.log('Answer:', data.answer);

  if (data.workflow_action?.type === 'UPDATE_CANVAS') {
    currentCanvas = data.workflow_action.payload;   // keep in sync
    updateNexAgentCanvas(data.workflow_action.payload);  // your UI function
  }
}
```

---

### 2. Ingest Documents
**Method:** `POST`  
**Path:** `/ingest`

Reads all `.pdf` and `.txt` files from the `data/` directory, chunks them, embeds them with `BAAI/bge-small-en`, and upserts into Pinecone. Idempotent.

#### Response
```json
{
  "status": "success",
  "files_ingested": { "questions.txt": 44 },
  "total_chunks": 44
}
```

---

### 3. Health Check
**Method:** `GET`  
**Path:** `/health`

```json
{
  "status": "ok",
  "agent": "langgraph-react",
  "llm": "llama-3.3-70b-versatile",
  "embeddings": "BAAI/bge-small-en",
  "version": "2.0"
}
```

---

### 4. Reset Session
**Method:** `DELETE`  
**Path:** `/reset-session/{session_id}`

Clears all conversation history for the given session so the user can start fresh.

```json
{ "status": "ok", "message": "Session 'user_abc123' has been reset." }
```

---

## Agent Tools (internal)

The LangGraph ReAct agent has access to 3 tools:

| Tool                     | When invoked                                                          |
|--------------------------|-----------------------------------------------------------------------|
| `update_workflow_canvas` | User asks to build, add, remove, or modify workflow nodes/config     |
| `search_nexagent_docs`   | User asks a factual question about NexAgent features or capabilities |
| `get_node_schema`        | Agent needs exact parameter names before building a workflow          |

---

## Example Interactions

### Build from scratch
```
User: "Create a workflow that reads topics from Google Sheets, asks Groq to write a blog post for each one, and emails the result"
→ Agent calls update_workflow_canvas with 4 nodes: ManualTrigger → GoogleSheets → Loop(Groq) → SendEmail
```

### Edit a specific node variable
```
User: "Change the email recipient in node n4 to reports@company.com"
Canvas: { nodes: [...n4 has to: "old@email.com"...], connections: [...] }
→ Agent updates only n4.config.to, returns full canvas with that change
```

### Add a node
```
User: "Add a Telegram notification after the Slack message"
Canvas: { nodes: [n1 Schedule, n2 Groq, n3 SlackMessage], connections: [...] }
→ Agent appends n4 TelegramSend, adds connection n3→n4, returns full updated canvas
```

### Ask a question (no canvas change)
```
User: "What's the difference between Groq and Gemini nodes?"
→ Agent calls search_nexagent_docs, returns text answer, workflow_action is null
```

---

## Key Points
- All requests use `POST` (except health = `GET`, reset = `DELETE`)
- Always set `Content-Type: application/json`
- Send `current_state` every turn so the agent knows what's on the canvas
- Use the same `session_id` across turns for conversation memory
- `workflow_action` is `null` for pure Q&A turns; only present when canvas changes


## Endpoints

### 1. Query Endpoint (Main)
**Method:** `POST`  
**Path:** `/query`

#### Request
```javascript
const response = await fetch('https://chatbot-backend-e5d4.onrender.com/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is DCS?'
  })
});

const data = await response.json();
```

#### Request Body
```json
{
  "question": "Your question here"
}
```

#### Response
```json
{
  "question": "What is DCS?",
  "answer": "Based on the provided context, DCS stands for Domestic Care Services...",
  "sources": ["domestic_care_services.pdf"]
}
```

#### Response Keys
- `question` - The question you asked
- `answer` - The AI-generated answer based on the PDF
- `sources` - List of documents used to generate the answer

---

### 

-

## React Native Expo Example

```javascript
import { useState } from 'react';
import { View, TextInput, Button, Text, ScrollView } from 'react-native';

export default function ChatScreen() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('https://chatbot-backend-e5d4.onrender.com/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question })
      });

      const data = await response.json();
      setAnswer(data.answer);
      setQuestion('');
    } catch (error) {
      console.error('Error:', error);
      setAnswer('Error fetching response');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <TextInput
        placeholder="Ask a question..."
        value={question}
        onChangeText={setQuestion}
        style={{ borderWidth: 1, padding: 10, marginBottom: 10 }}
      />
      <Button 
        title={loading ? 'Loading...' : 'Ask'} 
        onPress={askQuestion}
        disabled={loading}
      />
      <ScrollView style={{ marginTop: 20 }}>
        <Text>{answer}</Text>
      </ScrollView>
    </View>
  );
}
```

---

## Key Points
- All requests use `POST` method (except health check which is `GET`)
- Always set `Content-Type: application/json` header
- Response contains `answer` key with the AI response
- Response contains `sources` key with document references
- The API is ready to use - no setup needed on the client side

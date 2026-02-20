# RAG Chatbot API Documentation

## Base URL
```
https://chatbot-backend-e5d4.onrender.com
```

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

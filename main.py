from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from pinecone import Pinecone
from groq import Groq
from pypdf import PdfReader
import hashlib

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise RuntimeError("Missing required environment variables")

# Initialize Groq and Pinecone
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

# Helper function to get embeddings from Groq using text similarity
def get_simple_embedding(text: str):
    """Generate a simple embedding by hashing text"""
    # This is a placeholder - in production use a proper embedding model
    # For now, we'll use a simple hash-based approach
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert hex to float vector
    embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, len(hash_hex), 2)]
    # Pad to 384 dimensions (Pinecone index dimension)
    while len(embedding) < 384:
        embedding.extend(embedding[:384-len(embedding)])
    return embedding[:384]

# Ingest endpoint
@app.post("/ingest")
def ingest_pdf():
    try:
        pdf_path = "data/NexAgent_Documentation.pdf"
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF not found")
        
        logger.info(f"Loading PDF from {pdf_path}...")
        reader = PdfReader(pdf_path)
        
        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append({
                    "text": text,
                    "page": page_num + 1,
                    "source": "NexAgent_Documentation.pdf"
                })
        
        logger.info(f"Extracted {len(documents)} pages from PDF")
        
        # Chunk documents and create embeddings
        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = get_simple_embedding(doc["text"])
            vector_id = f"doc_{i}"
            vectors_to_upsert.append((
                vector_id,
                embedding,
                {
                    "text": doc["text"][:500],  # Store first 500 chars
                    "page": doc["page"],
                    "source": doc["source"]
                }
            ))
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Ingested {len(vectors_to_upsert)} vectors to Pinecone")
        
        return {
            "status": "success",
            "documents_ingested": len(documents),
            "vectors_created": len(vectors_to_upsert)
        }
    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query")
def query_chatbot(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")
        
        # Get embedding for question
        question_embedding = get_simple_embedding(request.question)
        
        # Query Pinecone for relevant documents
        query_result = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract context from Pinecone results
        context = "\n".join([
            match["metadata"].get("text", "")
            for match in query_result.get("matches", [])
            if match.get("score", 0) > 0.1
        ])
        
        if not context:
            context = "No relevant documents found in the knowledge base."
        
        # Step 1: Generate initial draft using Groq (Internal RAG Pass)
        draft_message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert technical assistant. Your job is to extract the EXACT step-by-step instructions and information from the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {request.question}\n\nProvide an accurate and complete answer based on the context. You MUST provide the actual information and instructions (e.g., exactly how to create an account, exactly what steps to click). Do not just say 'Follow the instructions in Section X'."
                }
            ],
            max_tokens=1024,
        )
        
        draft_answer = draft_message.choices[0].message.content
        
        # Step 2: Customer Support Pass (Summarization & Refinement)
        refinement_prompt = f"""You are a helpful, professional, and friendly customer support agent for NexAgent.
Your task is to take the following raw technical response and rewrite it to be clean and conversational. 
CRITICAL INSTRUCTIONS:
- IF the user's question was just a simple greeting (like "hi", "hello", "hey"), IGNORE the raw response entirely and simply greet them back warmly and ask how you can help.
- You MUST keep the final response short and easy to read. DO NOT exceed 15 lines of text under any circumstances.
- You MUST provide the actual detailed instructions, steps, and information to the user.
- DO NOT mention page numbers, section numbers, or document names (e.g., "Refer to section 2.1 (page 4)"). Just give the user the information directly.
- Use Markdown formatting (e.g., **bolding** key terms, using bullet points) to improve readability.
- Maintain a warm and polite tone.

User's Original Question: {request.question}

Raw Response to Rewrite:
{draft_answer}

Rewritten Customer Support Response:"""

        final_message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer support agent. Output only the final rewritten message without meta-commentary."
                },
                {
                    "role": "user",
                    "content": refinement_prompt
                }
            ],
            max_tokens=1024,
        )

        final_answer = final_message.choices[0].message.content
        
        # Extract sources
        sources = list(set([
            match["metadata"].get("source", "Unknown")
            for match in query_result.get("matches", [])
        ]))
        
        return QueryResponse(
            question=request.question,
            answer=final_answer,
            sources=sources
        )
    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

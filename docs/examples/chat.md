# Multi-Tenant Chat Example

Build a chat service with isolated user memories.

## Complete Implementation

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os

app = FastAPI(title="Memory-Enabled Chat Service")

# Shared configuration
base_config = {
    "llm_provider": "openai",
    "llm_api_key": os.getenv("OPENAI_API_KEY"),
    "llm_model": "gpt-4o-mini",
    "embedding_provider": "openai",
    "embedding_api_key": os.getenv("OPENAI_API_KEY"),
    "embedding_model": "text-embedding-3-small",
    "evolution_enabled": True,
    "decay_enabled": True,
    "decay_half_life_days": 30,
}

# Memory instances per user
memories = {}

def get_memory(user_id: str) -> GraphMem:
    """Get or create memory for a user."""
    if user_id not in memories:
        config = MemoryConfig(
            **base_config,
            turso_db_path=f"memories/{user_id}.db",
        )
        memories[user_id] = GraphMem(config, memory_id=f"chat_{user_id}", user_id=user_id)
    return memories[user_id]

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float
    remembered: bool

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Header(..., alias="X-User-ID"),
):
    memory = get_memory(user_id)
    
    # Store user message
    memory.ingest(
        f"User said: {request.message}",
        importance=MemoryImportance.MEDIUM,
    )
    
    # Query for context
    query_response = memory.query(request.message)
    
    # Generate response (simplified - you'd use your LLM here)
    response_text = generate_response(
        request.message,
        query_response.context,
    )
    
    # Store agent response
    memory.ingest(
        f"Agent responded: {response_text}",
        importance=MemoryImportance.LOW,
    )
    
    return ChatResponse(
        response=response_text,
        confidence=query_response.confidence,
        remembered=query_response.confidence > 0.5,
    )

@app.post("/learn")
async def learn(
    content: str,
    user_id: str = Header(..., alias="X-User-ID"),
    importance: str = "MEDIUM",
):
    """Explicitly teach the agent something."""
    memory = get_memory(user_id)
    
    importance_map = {
        "CRITICAL": MemoryImportance.CRITICAL,
        "HIGH": MemoryImportance.HIGH,
        "MEDIUM": MemoryImportance.MEDIUM,
        "LOW": MemoryImportance.LOW,
    }
    
    result = memory.ingest(
        content,
        importance=importance_map.get(importance, MemoryImportance.MEDIUM),
    )
    
    return {"status": "learned", "entities": result["entities"]}

@app.post("/forget")
async def forget(
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Clear user's memory (for testing or privacy)."""
    memory = get_memory(user_id)
    memory.clear()
    memories.pop(user_id, None)
    return {"status": "forgotten"}

@app.post("/consolidate")
async def consolidate(
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Consolidate user's memories."""
    memory = get_memory(user_id)
    events = memory.evolve()
    return {"events": len(events)}

@app.get("/memory-stats")
async def memory_stats(
    user_id: str = Header(..., alias="X-User-ID"),
):
    """Get user's memory statistics."""
    memory = get_memory(user_id)
    return memory.get_stats()

def generate_response(query: str, context: str) -> str:
    """Generate response using LLM. Replace with your implementation."""
    # This is where you'd call your LLM
    # For now, just return the context summary
    if context:
        return f"Based on what I remember: {context[:500]}..."
    return "I don't have any relevant memories about that."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Usage

```bash
# Start the server
python chat_service.py

# Learn something
curl -X POST "http://localhost:8000/learn" \
  -H "X-User-ID: alice" \
  -H "Content-Type: application/json" \
  -d '{"content": "My name is Alice and I work at Google"}'

# Chat
curl -X POST "http://localhost:8000/chat" \
  -H "X-User-ID: alice" \
  -H "Content-Type: application/json" \
  -d '{"message": "What do you know about me?"}'

# Response: "Based on what I remember: Your name is Alice and you work at Google..."
```

## Multi-User Isolation

```bash
# Alice's memory
curl -X POST "http://localhost:8000/learn" \
  -H "X-User-ID: alice" \
  -d '{"content": "I work at Google"}'

# Bob's memory
curl -X POST "http://localhost:8000/learn" \
  -H "X-User-ID: bob" \
  -d '{"content": "I work at Meta"}'

# Alice can't see Bob's data
curl -X POST "http://localhost:8000/chat" \
  -H "X-User-ID: alice" \
  -d '{"message": "Where does Bob work?"}'
# Response: "I don't have any relevant memories about that."
```


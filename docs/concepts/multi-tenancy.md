# Multi-Tenancy

GraphMem provides complete data isolation between users.

## The Problem

In production systems:
- Multiple users share the same infrastructure
- Each user's data must be completely private
- Cross-tenant data leakage is unacceptable

## The Solution

Every piece of data in GraphMem is tagged with:
- `user_id` - The user who owns the data
- `memory_id` - The memory context (session, project, etc.)

---

## Basic Usage

```python
from graphmem import GraphMem, MemoryConfig

# Shared configuration
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
)

# Alice's memory
alice = GraphMem(config, user_id="alice", memory_id="chat")
alice.ingest("I work at Google as a senior engineer")

# Bob's memory (completely isolated)
bob = GraphMem(config, user_id="bob", memory_id="chat")
bob.ingest("I'm a doctor at Mayo Clinic")

# Alice can NEVER see Bob's data
alice.query("What does Bob do?")  # → "No information found"

# Bob can NEVER see Alice's data
bob.query("Where does Alice work?")  # → "No information found"
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Neo4j / Turso / InMemory                           │
├────────────────────────────────────┬─────────────────────────────────────┤
│           USER: alice              │            USER: bob                 │
│  ┌─────────────────────────────┐   │   ┌─────────────────────────────┐   │
│  │ 🏢 Google  → 👤 Alice       │   │   │ 🏥 Mayo Clinic → 👤 Bob     │   │
│  │     ↓                       │   │   │       ↓                     │   │
│  │ 💼 Senior Engineer          │   │   │   🩺 Doctor                 │   │
│  └─────────────────────────────┘   │   └─────────────────────────────┘   │
├────────────────────────────────────┴─────────────────────────────────────┤
│                    Redis Cache (Also Isolated by user_id)                 │
│  alice:query:*  alice:search:*     │     bob:query:*  bob:search:*       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## How Isolation Works

### Data Storage

All nodes and edges are tagged with `user_id`:

```python
# When Alice ingests
node = MemoryNode(
    id="google_001",
    name="Google",
    user_id="alice",      # ← Tagged
    memory_id="chat",     # ← Tagged
)
```

### Query Filtering

All queries are filtered by `user_id`:

```sql
-- Neo4j / Turso
SELECT * FROM entities 
WHERE user_id = 'alice' 
  AND memory_id = 'chat'
  AND embedding <-> query_embedding < threshold
```

### Cache Isolation

Redis keys include `user_id`:

```
query:alice:chat:abc123   # Alice's cache
query:bob:chat:xyz789     # Bob's cache
```

---

## Memory Scoping

Users can have multiple memory contexts:

```python
# Alice has separate memories for different contexts
alice_chat = GraphMem(config, user_id="alice", memory_id="chat")
alice_notes = GraphMem(config, user_id="alice", memory_id="notes")
alice_work = GraphMem(config, user_id="alice", memory_id="work")

# Each is isolated
alice_chat.ingest("Chatbot conversation...")
alice_notes.ingest("Personal notes...")
alice_work.ingest("Work documents...")

# Queries only search within the specific memory
alice_chat.query("What did we discuss?")  # Only searches chat
```

---

## Multi-Tenant Service Pattern

```python
from graphmem import GraphMem, MemoryConfig

class MemoryService:
    def __init__(self):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o-mini",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            neo4j_uri="neo4j+s://...",
            neo4j_password="...",
            redis_url="redis://...",
        )
        self._memories = {}
    
    def get_memory(self, user_id: str, memory_id: str) -> GraphMem:
        """Get or create memory for a user/context."""
        key = f"{user_id}:{memory_id}"
        if key not in self._memories:
            self._memories[key] = GraphMem(
                self.config,
                user_id=user_id,
                memory_id=memory_id,
            )
        return self._memories[key]
    
    def ingest(self, user_id: str, memory_id: str, content: str):
        memory = self.get_memory(user_id, memory_id)
        return memory.ingest(content)
    
    def query(self, user_id: str, memory_id: str, question: str):
        memory = self.get_memory(user_id, memory_id)
        return memory.query(question)

# FastAPI Example
from fastapi import FastAPI, Depends, Header

app = FastAPI()
service = MemoryService()

@app.post("/ingest")
async def ingest(
    content: str,
    user_id: str = Header(...),
    memory_id: str = Header(default="default"),
):
    result = service.ingest(user_id, memory_id, content)
    return {"status": "success", "entities": result["entities"]}

@app.post("/query")
async def query(
    question: str,
    user_id: str = Header(...),
    memory_id: str = Header(default="default"),
):
    response = service.query(user_id, memory_id, question)
    return {"answer": response.answer, "confidence": response.confidence}
```

---

## Per-User Database Files (Turso)

For maximum isolation, use separate database files:

```python
def get_user_memory(user_id: str) -> GraphMem:
    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
        
        # Each user gets their own file
        turso_db_path=f"memories/{user_id}.db",
    )
    return GraphMem(config, user_id=user_id)

# Alice's data in memories/alice.db
# Bob's data in memories/bob.db
```

---

## Cache Invalidation

When data changes, only that user's cache is invalidated:

```python
# Alice ingests new data
alice.ingest("New information")

# Only Alice's cache is cleared
# Bob's cache remains intact
```

This is handled automatically—you don't need to manage it.

---

## Security Considerations

1. **Always require authentication** - Validate `user_id` before any operation
2. **Never trust client-provided user_id** - Extract from verified auth tokens
3. **Audit access patterns** - Log who queries what
4. **Consider data residency** - Use region-specific storage for compliance

```python
# Example: Extract user_id from JWT
from fastapi import Depends
from jose import jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload["sub"]  # user_id

@app.post("/query")
async def query(question: str, user_id: str = Depends(get_current_user)):
    # user_id is verified, not client-provided
    memory = service.get_memory(user_id, "default")
    return memory.query(question)
```


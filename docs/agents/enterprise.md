# Enterprise Agents

Build multi-tenant agents for production environments.

## Pattern Overview

An enterprise agent:
- Serves multiple customers with isolated data
- Handles policy updates and conflict resolution
- Scales horizontally
- Maintains audit trails

---

## Multi-Tenant Agent

```python
from graphmem import GraphMem, MemoryConfig

class MultiTenantAgent:
    """Agent that serves multiple customers with isolated memories."""
    
    def __init__(self, base_config: dict):
        self.base_config = base_config
        self.memories = {}  # tenant_id -> GraphMem
    
    def get_memory(self, tenant_id: str) -> GraphMem:
        """Get or create memory for a tenant."""
        if tenant_id not in self.memories:
            config = MemoryConfig(
                **self.base_config,
                user_id=tenant_id,
                memory_id=f"tenant_{tenant_id}",
                turso_db_path=f"tenants/{tenant_id}/memory.db",
            )
            self.memories[tenant_id] = GraphMem(config)
        return self.memories[tenant_id]
    
    def ingest(self, tenant_id: str, content: str):
        """Ingest content for a specific tenant."""
        memory = self.get_memory(tenant_id)
        return memory.ingest(content)
    
    def query(self, tenant_id: str, question: str) -> str:
        """Query a specific tenant's memory."""
        memory = self.get_memory(tenant_id)
        response = memory.query(question)
        return response.answer

# Usage
base_config = {
    "llm_provider": "openai",
    "llm_api_key": "sk-...",
    "llm_model": "gpt-4o-mini",
    "embedding_provider": "openai",
    "embedding_api_key": "sk-...",
    "embedding_model": "text-embedding-3-small",
}

agent = MultiTenantAgent(base_config)

# Each tenant's data is completely isolated
agent.ingest("acme", "Acme's product costs $99.")
agent.ingest("globex", "Globex's product costs $149.")

print(agent.query("acme", "What's the product price?"))   # → "$99"
print(agent.query("globex", "What's the product price?")) # → "$149"
```

---

## Customer Support Agent

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance

class SupportAgent:
    """Agent for customer support with policy updates."""
    
    def __init__(self, company_id: str):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            turso_db_path=f"support/{company_id}.db",
            evolution_enabled=True,
            decay_enabled=True,
        )
        self.memory = GraphMem(self.config)
    
    def update_policy(self, content: str, effective_date: str):
        """Update knowledge base with new policy."""
        enriched = f"""
        POLICY UPDATE (Effective: {effective_date})
        {content}
        """
        self.memory.ingest(
            enriched,
            importance=MemoryImportance.CRITICAL,
        )
        # Resolve conflicts with old policies
        self.memory.evolve()
    
    def answer_ticket(self, ticket: str) -> dict:
        """Answer a support ticket."""
        response = self.memory.query(ticket)
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "needs_escalation": response.confidence < 0.6,
        }

# Usage
agent = SupportAgent("acme_corp")

# Initial policy
agent.update_policy(
    "Returns accepted within 30 days. Refunds in 5-7 days.",
    "2023-01-01"
)

# Policy update - old one automatically superseded
agent.update_policy(
    "Returns accepted within 60 days. Refunds in 2-3 days.",
    "2024-01-01"
)

# Agent uses current policy
result = agent.answer_ticket("What's your return policy?")
# → "Returns within 60 days, refunds in 2-3 days"
```

---

## Enterprise Knowledge Base

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance
from datetime import datetime

class EnterpriseKB:
    """Enterprise knowledge base with audit trail."""
    
    def __init__(self, org_id: str):
        self.config = MemoryConfig(
            llm_provider="azure",
            llm_api_key="...",
            azure_endpoint="https://...",
            azure_deployment="gpt-4",
            llm_model="gpt-4",
            embedding_provider="azure",
            embedding_api_key="...",
            azure_embedding_deployment="text-embedding-ada-002",
            embedding_model="text-embedding-ada-002",
            neo4j_uri="neo4j+s://...",
            neo4j_password="...",
            redis_url="redis://...",
            user_id=org_id,
            evolution_enabled=True,
        )
        self.memory = GraphMem(self.config)
        self.org_id = org_id
    
    def add_policy(self, name: str, content: str, effective: str):
        """Add or update a policy document."""
        enriched = f"""
        POLICY: {name}
        Effective Date: {effective}
        Last Updated: {datetime.now().isoformat()}
        
        {content}
        """
        self.memory.ingest(
            enriched,
            metadata={"type": "policy", "name": name},
            importance=MemoryImportance.CRITICAL,
        )
        self.memory.evolve()
    
    def add_procedure(self, name: str, steps: list[str]):
        """Add a procedure manual."""
        content = f"PROCEDURE: {name}\n\n"
        content += "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        
        self.memory.ingest(
            content,
            metadata={"type": "procedure", "name": name},
            importance=MemoryImportance.HIGH,
        )
    
    def add_employee(self, data: dict):
        """Add employee information."""
        content = f"""
        EMPLOYEE: {data['name']}
        Title: {data['title']}
        Department: {data['department']}
        Email: {data['email']}
        Reports to: {data.get('manager', 'N/A')}
        Start date: {data.get('start_date', 'N/A')}
        """
        self.memory.ingest(
            content,
            metadata={"type": "employee", **data},
            importance=MemoryImportance.MEDIUM,
        )
    
    def answer(self, question: str, department: str = None) -> dict:
        """Answer a question from the knowledge base."""
        if department:
            question = f"[{department}] {question}"
        
        response = self.memory.query(question)
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [n.name for n in response.nodes[:3]],
            "escalate": response.confidence < 0.5,
        }
    
    def bulk_import(self, documents: list[dict]):
        """Bulk import documents."""
        self.memory.ingest_batch(
            documents,
            max_workers=20,
            aggressive=True,
        )
        self.memory.evolve()
```

---

## FastAPI Production Service

```python
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import os

app = FastAPI(title="Enterprise Memory Service")

# Configuration from environment
config = MemoryConfig(
    llm_provider="azure",
    llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    llm_model="gpt-4",
    embedding_provider="azure",
    embedding_api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_embedding_deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    embedding_model="text-embedding-ada-002",
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    redis_url=os.getenv("REDIS_URL"),
)

# Tenant memories
memories = {}

def get_memory(tenant_id: str = Header(..., alias="X-Tenant-ID")) -> GraphMem:
    if tenant_id not in memories:
        memories[tenant_id] = GraphMem(
            config,
            user_id=tenant_id,
            memory_id="kb",
        )
    return memories[tenant_id]

class IngestRequest(BaseModel):
    content: str
    importance: str = "MEDIUM"

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest(
    request: IngestRequest,
    memory: GraphMem = Depends(get_memory),
):
    importance_map = {
        "CRITICAL": MemoryImportance.CRITICAL,
        "HIGH": MemoryImportance.HIGH,
        "MEDIUM": MemoryImportance.MEDIUM,
        "LOW": MemoryImportance.LOW,
    }
    
    result = memory.ingest(
        request.content,
        importance=importance_map.get(request.importance),
    )
    return {"status": "success", "entities": result["entities"]}

@app.post("/query")
async def query(
    request: QueryRequest,
    memory: GraphMem = Depends(get_memory),
):
    response = memory.query(request.question)
    return {
        "answer": response.answer,
        "confidence": response.confidence,
    }

@app.post("/evolve")
async def evolve(memory: GraphMem = Depends(get_memory)):
    events = memory.evolve()
    return {"events": len(events)}

@app.get("/stats")
async def stats(memory: GraphMem = Depends(get_memory)):
    return memory.get_stats()
```

---

## Best Practices

1. **Isolate by tenant** - Use `user_id` for complete separation
2. **Use enterprise storage** - Neo4j + Redis for scale
3. **Mark policy as critical** - Should never decay
4. **Evolve after updates** - Resolve conflicts immediately
5. **Audit everything** - Include timestamps in metadata
6. **Use Azure OpenAI** - Enterprise compliance
7. **Set escalation thresholds** - Route low-confidence answers to humans


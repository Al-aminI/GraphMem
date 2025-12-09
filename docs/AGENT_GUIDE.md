# 🧠 Building AI Agents with GraphMem

> **The definitive guide to building production AI agents with human-like memory**

GraphMem provides the memory layer that transforms stateless LLMs into intelligent agents capable of learning, remembering, and evolving over time.

---

## Table of Contents

1. [Why GraphMem for Agents?](#why-graphmem-for-agents)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Agent Patterns](#agent-patterns)
5. [Production Architecture](#production-architecture)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Complete Examples](#complete-examples)

---

## Why GraphMem for Agents?

### The Problem with Current Agent Memory

| Approach | Problem |
|----------|---------|
| **Context stuffing** | Token limits, expensive, no learning |
| **Simple vector DB** | No relationships, no temporal awareness |
| **Key-value stores** | No semantic understanding |
| **Traditional RAG** | Retrieves chunks, not knowledge |

### How GraphMem Solves This

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPHMEM MEMORY SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📥 INGEST           🧠 UNDERSTAND           📤 RETRIEVE        │
│  ─────────           ───────────             ──────────         │
│  • Text              • Entity extraction     • Semantic search  │
│  • Documents         • Relationship mapping  • Graph traversal  │
│  • Conversations     • Alias resolution      • Community query  │
│                      • Temporal tracking     • Multi-hop reason │
│                                                                  │
│  🔄 EVOLVE           ⚡ OPTIMIZE             🔒 ISOLATE         │
│  ────────            ──────────              ─────────          │
│  • Consolidation     • PageRank scoring      • Multi-tenant     │
│  • Decay old info    • Importance weighting  • User isolation   │
│  • Conflict resolve  • Caching (Redis)       • Session scoping  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Core (in-memory only - for testing)
pip install agentic-graph-mem

# With persistence (RECOMMENDED for production)
pip install "agentic-graph-mem[libsql]"
```

### Your First Memory-Enabled Agent

!!! warning "Important: Enable Persistence!"
    Without `turso_db_path`, your agent forgets everything on restart!

```python
from graphmem import GraphMem, MemoryConfig

# Initialize with your LLM provider
config = MemoryConfig(
    # LLM for extraction and querying
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    
    # Embeddings for semantic search
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ CRITICAL: Persistent storage!
    turso_db_path="agent_memory.db",
    
    # Enable self-evolution
    evolution_enabled=True,
    auto_evolve=True,  # Evolve after each ingestion
)

# Create memory instance with a consistent memory_id
memory = GraphMem(config, memory_id="my_agent")

# Your agent can now remember!
class SimpleAgent:
    def __init__(self, memory: GraphMem):
        self.memory = memory
    
    def learn(self, information: str):
        """Agent learns new information"""
        self.memory.ingest(information)
    
    def ask(self, question: str) -> str:
        """Agent answers from memory"""
        response = self.memory.query(question)
        return response.answer
    
    def reflect(self):
        """Agent consolidates and improves memory"""
        self.memory.evolve()

# Use the agent
agent = SimpleAgent(memory)
agent.learn("Tesla was founded in 2003. Elon Musk became CEO in 2008.")
agent.learn("SpaceX was founded by Elon Musk in 2002.")

print(agent.ask("What companies did Elon Musk found?"))
# → "Elon Musk founded SpaceX in 2002 and became CEO of Tesla in 2008."

agent.reflect()  # Consolidate knowledge about Elon Musk
```

---

## Core Concepts

### 1. The Three Pillars: Ingest → Query → Evolve

```python
# INGEST: Feed information to memory
memory.ingest("""
    Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei.
    They previously worked at OpenAI. Anthropic created Claude.
""")

# QUERY: Ask questions
response = memory.query("Who founded Anthropic?")
print(response.answer)      # "Dario Amodei and Daniela Amodei founded Anthropic in 2021"
print(response.confidence)  # 0.95
print(response.context)     # Full context used for answering

# EVOLVE: Improve memory over time
events = memory.evolve()
for event in events:
    print(f"{event.evolution_type}: {event.description}")
    # CONSOLIDATION: Merged 3 mentions of "Anthropic" into 1 entity
    # DECAY: Archived 2 outdated relationships
```

### 2. Knowledge Graph Structure

GraphMem automatically builds a knowledge graph from your text:

```
                    ┌─────────────┐
                    │   Anthropic │
                    │ (Company)   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │Dario Amodei │ │Daniela      │ │   Claude    │
    │  (Person)   │ │Amodei       │ │  (Product)  │
    │             │ │(Person)     │ │             │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │
           └───────┬───────┘
                   │
                   ▼
            ┌─────────────┐
            │   OpenAI    │
            │ (Company)   │
            └─────────────┘

Relationships:
• Dario Amodei --[founded]--> Anthropic (2021-present)
• Dario Amodei --[worked_at]--> OpenAI (past)
• Anthropic --[created]--> Claude
```

### 3. Exhaustive Extraction

GraphMem extracts **everything** from your text:

```python
memory.ingest("""
    In Q3 2024, Nvidia (NVDA) reported $35.1B revenue, up 94% YoY.
    CEO Jensen Huang announced Blackwell B200 shipping in early 2025.
    The company has 26,000 employees worldwide.
""")

# GraphMem extracts:
# ENTITIES:
# - Nvidia (Company) [aliases: NVDA, Nvidia Corporation]
# - Jensen Huang (Person) [aliases: J. Huang]
# - $35.1B (Amount)
# - 94% (Percentage)
# - Q3 2024 (Date)
# - Blackwell B200 (Product)
# - 26,000 (Number)
# - early 2025 (Date)

# RELATIONSHIPS:
# - Jensen Huang --[is CEO of]--> Nvidia
# - Nvidia --[reported revenue]--> $35.1B [valid: Q3 2024]
# - Nvidia --[achieved growth]--> 94%
# - Nvidia --[has employees]--> 26,000
# - Blackwell B200 --[ships in]--> early 2025
```

### 4. Source Chunk Preservation

Every entity preserves its **original source text** for accurate answers:

```python
# When you query, GraphMem provides:
response = memory.query("What is Nvidia's revenue?")

# The LLM sees:
# 1. ORIGINAL SOURCE TEXT (full context)
# 2. EXTRACTED ENTITIES (structured)
# 3. RELATIONSHIPS (connections)
# 4. COMMUNITY SUMMARIES (high-level understanding)
```

---

## Agent Patterns

### Pattern 1: Conversational Agent with Persistent Memory

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance

class ConversationalAgent:
    """Agent that remembers conversations across sessions."""
    
    def __init__(self, user_id: str):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o-mini",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            
            # Persist to SQLite
            turso_db_path=f"memories/{user_id}.db",
            
            # User isolation
            user_id=user_id,
            
            # Evolution settings
            evolution_enabled=True,
            decay_enabled=True,
            decay_half_life_days=30,  # Forget unused info after ~30 days
        )
        self.memory = GraphMem(self.config)
        self.user_id = user_id
    
    def chat(self, user_message: str) -> str:
        """Process user message and generate response."""
        
        # 1. Store the user's message as a memory
        self.memory.ingest(
            f"User said: {user_message}",
            metadata={"type": "user_message", "timestamp": "now"},
            importance=MemoryImportance.MEDIUM,
        )
        
        # 2. Query memory for relevant context
        response = self.memory.query(user_message)
        
        # 3. Generate response (you'd use your own LLM here)
        agent_response = self._generate_response(user_message, response.context)
        
        # 4. Store the agent's response too
        self.memory.ingest(
            f"Agent responded: {agent_response}",
            metadata={"type": "agent_response"},
            importance=MemoryImportance.LOW,
        )
        
        return agent_response
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with memory context."""
        # Your LLM call here
        pass
    
    def end_session(self):
        """Consolidate memories at end of session."""
        self.memory.evolve()

# Usage
agent = ConversationalAgent(user_id="user_123")
agent.chat("My name is Alice and I work at Google.")
agent.chat("I'm interested in machine learning.")

# Later session...
agent.chat("What do you know about me?")
# → "You're Alice, you work at Google, and you're interested in machine learning."
```

### Pattern 2: Research Agent with Multi-Document Learning

```python
class ResearchAgent:
    """Agent that learns from multiple documents and answers complex questions."""
    
    def __init__(self):
        self.config = MemoryConfig(
            llm_provider="azure_openai",
            llm_api_key="...",
            llm_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
            azure_deployment="gpt-4",
            llm_model="gpt-4",
            azure_api_version="2024-02-15-preview",
            
            embedding_provider="azure_openai",
            embedding_api_key="...",
            embedding_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
            azure_embedding_deployment="text-embedding-ada-002",
            embedding_model="text-embedding-ada-002",
            
            # Use Neo4j for production graph storage
            neo4j_uri="neo4j+s://your-instance.databases.neo4j.io",
            neo4j_username="neo4j",
            neo4j_password="...",
            
            # Redis for caching
            redis_url="redis://...",
            
            # Aggressive evolution for research
            evolution_enabled=True,
            consolidation_threshold=0.75,  # More aggressive merging
        )
        self.memory = GraphMem(self.config)
    
    def ingest_documents(self, documents: list[dict]):
        """Ingest multiple documents efficiently."""
        result = self.memory.ingest_batch(
            documents,
            max_workers=20,  # Parallel processing
            aggressive=True,
            show_progress=True,
        )
        print(f"Ingested {result['documents_processed']} docs")
        print(f"Extracted {result['total_entities']} entities")
        print(f"Found {result['total_relationships']} relationships")
        
        # Evolve after batch ingestion
        self.memory.evolve()
    
    def research(self, question: str) -> dict:
        """Answer complex research questions."""
        response = self.memory.query(question)
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [n.name for n in response.nodes[:5]],
            "related_entities": [n.name for n in response.nodes],
            "context_tokens": len(response.context.split()),
        }
    
    def find_connections(self, entity_a: str, entity_b: str) -> str:
        """Find how two entities are connected."""
        query = f"How are {entity_a} and {entity_b} related or connected?"
        response = self.memory.query(query)
        return response.answer

# Usage
agent = ResearchAgent()

# Ingest research papers
papers = [
    {"id": "paper1", "content": "...paper about transformers..."},
    {"id": "paper2", "content": "...paper about attention mechanisms..."},
    {"id": "paper3", "content": "...paper about GPT architecture..."},
]
agent.ingest_documents(papers)

# Ask complex questions
result = agent.research("How did attention mechanisms evolve into modern LLMs?")
print(result["answer"])

# Find entity connections
print(agent.find_connections("Transformers", "GPT-4"))
```

### Pattern 3: Customer Support Agent with Conflict Resolution

```python
class SupportAgent:
    """Agent that handles customer support with up-to-date knowledge."""
    
    def __init__(self, company_id: str):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            
            turso_db_path=f"support/{company_id}.db",
            
            # Critical: Enable temporal validity for policy updates
            evolution_enabled=True,
            decay_enabled=True,
        )
        self.memory = GraphMem(self.config)
    
    def update_knowledge(self, content: str, importance: str = "HIGH"):
        """Update knowledge base with new information."""
        from graphmem import MemoryImportance
        
        importance_map = {
            "CRITICAL": MemoryImportance.CRITICAL,
            "HIGH": MemoryImportance.HIGH,
            "MEDIUM": MemoryImportance.MEDIUM,
            "LOW": MemoryImportance.LOW,
        }
        
        self.memory.ingest(
            content,
            importance=importance_map.get(importance, MemoryImportance.MEDIUM),
        )
        
        # CRITICAL: Evolve to resolve conflicts with old information
        # This uses priority-based decay to supersede outdated facts
        self.memory.evolve()
    
    def answer_ticket(self, ticket: str) -> dict:
        """Answer a support ticket using knowledge base."""
        response = self.memory.query(ticket)
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "needs_escalation": response.confidence < 0.6,
        }

# Usage
agent = SupportAgent("acme_corp")

# Initial knowledge
agent.update_knowledge("""
    Our return policy allows returns within 30 days of purchase.
    Refunds are processed within 5-7 business days.
""")

# Policy update - GraphMem will resolve the conflict!
agent.update_knowledge("""
    POLICY UPDATE (2024): Our new return policy allows returns within 60 days.
    Refunds are now processed within 2-3 business days.
""", importance="CRITICAL")

# Agent now answers with updated info
result = agent.answer_ticket("What's your return policy?")
print(result["answer"])
# → "Our return policy allows returns within 60 days. Refunds are processed within 2-3 business days."
# (Old 30-day policy was automatically superseded)
```

### Pattern 4: Multi-Tenant SaaS Agent

```python
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
                
                # CRITICAL: Tenant isolation
                user_id=tenant_id,
                memory_id=f"tenant_{tenant_id}",
                
                # Each tenant gets own database
                turso_db_path=f"tenants/{tenant_id}/memory.db",
            )
            self.memories[tenant_id] = GraphMem(config)
        
        return self.memories[tenant_id]
    
    def ingest(self, tenant_id: str, content: str):
        """Ingest content for a specific tenant."""
        memory = self.get_memory(tenant_id)
        memory.ingest(content)
    
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

## Production Architecture

### Recommended Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    APPLICATION LAYER                      │    │
│  │                                                          │    │
│  │   FastAPI / Flask / Django                               │    │
│  │       ↓                                                  │    │
│  │   GraphMem Instance (per request or singleton)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     STORAGE LAYER                         │    │
│  │                                                          │    │
│  │  Neo4j Aura (Graph)  ←→  Redis Cloud (Cache)            │    │
│  │         ↑                        ↑                       │    │
│  │         └────────────────────────┘                       │    │
│  │                    ↓                                     │    │
│  │            Turso (Backup/Vectors)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LLM PROVIDERS                          │    │
│  │                                                          │    │
│  │  OpenAI  |  Azure OpenAI  |  Anthropic  |  Local LLMs   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### FastAPI Integration Example

```python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from graphmem import GraphMem, MemoryConfig
from functools import lru_cache

app = FastAPI()

# Singleton memory instance
@lru_cache()
def get_memory() -> GraphMem:
    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_model="text-embedding-3-small",
        
        # Production storage
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        
        # Redis caching
        redis_url=os.getenv("REDIS_URL"),
        
        evolution_enabled=True,
    )
    return GraphMem(config)

class IngestRequest(BaseModel):
    content: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

@app.post("/ingest")
async def ingest(request: IngestRequest, memory: GraphMem = Depends(get_memory)):
    result = memory.ingest(request.content, metadata=request.metadata)
    return {"status": "success", "entities": result["entities"]}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, memory: GraphMem = Depends(get_memory)):
    response = memory.query(request.question)
    return QueryResponse(
        answer=response.answer,
        confidence=response.confidence,
        sources=[n.name for n in response.nodes[:5]],
    )

@app.post("/evolve")
async def evolve(memory: GraphMem = Depends(get_memory)):
    events = memory.evolve()
    return {"events": len(events)}

@app.get("/stats")
async def stats(memory: GraphMem = Depends(get_memory)):
    return memory.get_stats()
```

### High-Throughput Batch Ingestion

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class HighThroughputIngestion:
    """Ingest millions of documents efficiently."""
    
    def __init__(self, memory: GraphMem):
        self.memory = memory
    
    def ingest_large_dataset(
        self,
        documents: list[dict],
        batch_size: int = 1000,
        max_workers: int = 20,
    ):
        """Ingest documents in batches with progress tracking."""
        total = len(documents)
        processed = 0
        
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            
            result = self.memory.ingest_batch(
                batch,
                max_workers=max_workers,
                aggressive=True,
                show_progress=True,
                rebuild_communities=False,  # Defer until end
            )
            
            processed += result["documents_processed"]
            print(f"Progress: {processed}/{total} ({100*processed/total:.1f}%)")
        
        # Rebuild communities once at the end
        print("Building communities...")
        self.memory.evolve()
        
        return {"total_processed": processed}

# Usage
ingestion = HighThroughputIngestion(memory)
ingestion.ingest_large_dataset(
    documents=large_dataset,  # Millions of docs
    batch_size=1000,
    max_workers=20,
)
```

---

## Advanced Features

### 1. Temporal Queries (Point-in-Time)

```python
from datetime import datetime

# Ingest historical data
memory.ingest("""
    Steve Jobs was CEO of Apple from 1997 to 2011.
    Tim Cook became CEO in August 2011 and is still CEO today.
""")

# Query about the past
response = memory.query("Who was CEO of Apple in 2005?")
print(response.answer)  # → "Steve Jobs was CEO of Apple from 1997 to 2011"

# Query about the present
response = memory.query("Who is the current CEO of Apple?")
print(response.answer)  # → "Tim Cook has been CEO since August 2011"
```

### 2. Alias-Aware Retrieval

```python
# GraphMem automatically extracts and uses aliases
memory.ingest("""
    Dr. Alexander Chen, also known as "The Quantum Pioneer", 
    founded Quantum AI Labs. Alex Chen received his PhD from MIT.
""")

# All of these work:
memory.query("What did Dr. Chen do?")
memory.query("Who is Alexander Chen?")
memory.query("Tell me about The Quantum Pioneer")
memory.query("What did Alex Chen study?")
# All return information about the same person!
```

### 3. Multi-Hop Reasoning

```python
# GraphMem can traverse relationships to answer complex questions
memory.ingest("Apple was founded by Steve Jobs.")
memory.ingest("Steve Jobs also founded NeXT.")
memory.ingest("NeXT was acquired by Apple in 1997.")
memory.ingest("Tim Cook worked at Compaq before Apple.")

# Multi-hop question
response = memory.query("What's the connection between NeXT and Tim Cook?")
# GraphMem traverses: NeXT → Steve Jobs → Apple → Tim Cook
print(response.answer)
# → "NeXT was founded by Steve Jobs and acquired by Apple in 1997. 
#    Tim Cook currently works at Apple as CEO."
```

### 4. Conflict Resolution via Evolution

```python
# Initial fact
memory.ingest("The company has 1,000 employees.")

# Updated fact
memory.ingest("BREAKING: The company now has 2,500 employees after expansion.")

# Evolution automatically resolves conflicts
memory.evolve()

# Queries return the newer information
response = memory.query("How many employees does the company have?")
print(response.answer)  # → "2,500 employees (after expansion)"
```

### 5. Importance-Based Memory

```python
from graphmem import MemoryImportance

# Critical information (never decays)
memory.ingest(
    "Customer's allergies: peanuts, shellfish",
    importance=MemoryImportance.CRITICAL,
)

# Normal information (may decay over time)
memory.ingest(
    "Customer mentioned they like coffee",
    importance=MemoryImportance.LOW,
)

# After evolution, low-importance unused memories decay
# Critical memories are always preserved
```

---

## Best Practices

### 1. Structure Your Ingestion

```python
# ❌ BAD: Unstructured dumps
memory.ingest("stuff happened today blah blah...")

# ✅ GOOD: Clear, factual statements
memory.ingest("""
    On January 15, 2024, Acme Corp announced Q4 earnings:
    - Revenue: $5.2 billion (up 23% YoY)
    - Net income: $890 million
    - CEO Jane Smith attributed growth to AI products
""")
```

### 2. Use Batch Ingestion for Large Datasets

```python
# ❌ BAD: Sequential ingestion
for doc in documents:
    memory.ingest(doc["content"])  # Slow!

# ✅ GOOD: Batch ingestion
memory.ingest_batch(
    documents,
    max_workers=20,
    aggressive=True,
)
```

### 3. Evolve Regularly

```python
# Option 1: Auto-evolve (simple)
config = MemoryConfig(..., auto_evolve=True)

# Option 2: Explicit evolution (more control)
# After ingestion sessions
memory.ingest_batch(docs)
memory.evolve()

# Or on a schedule (e.g., daily)
import schedule
schedule.every().day.at("02:00").do(memory.evolve)
```

### 4. Use Appropriate Storage

| Use Case | Recommended Storage |
|----------|-------------------|
| **Development/Testing** | Turso (SQLite) - zero setup |
| **Single-server production** | Turso + Redis |
| **Multi-server production** | Neo4j + Redis |
| **Enterprise/High-scale** | Neo4j Aura + Redis Cloud |

### 5. Handle Errors Gracefully

```python
from graphmem.core.exceptions import IngestionError, QueryError

try:
    memory.ingest(content)
except IngestionError as e:
    logger.error(f"Ingestion failed: {e}")
    # Retry or queue for later

try:
    response = memory.query(question)
except QueryError as e:
    logger.error(f"Query failed: {e}")
    # Fallback to default response
```

---

## Complete Examples

### Example 1: Personal Knowledge Assistant

```python
"""
A personal knowledge assistant that learns from your notes,
articles, and conversations.
"""

from graphmem import GraphMem, MemoryConfig, MemoryImportance
import os

class PersonalAssistant:
    def __init__(self, user_name: str):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model="gpt-4o-mini",
            embedding_provider="openai",
            embedding_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model="text-embedding-3-small",
            
            # Persistent personal database
            turso_db_path=f"~/.assistant/{user_name}.db",
            user_id=user_name,
            
            # Memory evolution
            evolution_enabled=True,
            decay_enabled=True,
            decay_half_life_days=90,  # Keep memories for ~3 months
        )
        self.memory = GraphMem(self.config)
        self.user_name = user_name
    
    def save_note(self, note: str, tags: list[str] = None):
        """Save a personal note."""
        self.memory.ingest(
            note,
            metadata={"type": "note", "tags": tags or []},
            importance=MemoryImportance.MEDIUM,
        )
    
    def save_article(self, title: str, content: str, url: str = None):
        """Save an article with high importance."""
        self.memory.ingest(
            f"Article: {title}\n\n{content}",
            metadata={"type": "article", "url": url},
            importance=MemoryImportance.HIGH,
        )
    
    def remember_fact(self, fact: str):
        """Save an important fact."""
        self.memory.ingest(
            fact,
            importance=MemoryImportance.VERY_HIGH,
        )
    
    def ask(self, question: str) -> str:
        """Ask your assistant anything."""
        response = self.memory.query(question)
        return response.answer
    
    def daily_digest(self):
        """Get a summary of what you learned recently."""
        response = self.memory.query(
            "What are the most important things I learned recently?"
        )
        return response.answer
    
    def consolidate(self):
        """Run weekly to consolidate memories."""
        self.memory.evolve()

# Usage
assistant = PersonalAssistant("alice")

# Learning
assistant.save_note("Meeting with Bob: Discuss Q1 roadmap next Tuesday")
assistant.save_article(
    "The Future of AI Agents",
    "AI agents are becoming more capable...",
    url="https://example.com/ai-agents"
)
assistant.remember_fact("My AWS account ID is 123456789")

# Querying
print(assistant.ask("When is my meeting with Bob?"))
print(assistant.ask("What's my AWS account ID?"))
print(assistant.daily_digest())

# Weekly consolidation
assistant.consolidate()
```

### Example 2: Enterprise Knowledge Base Agent

```python
"""
Enterprise knowledge base that handles:
- Policy documents
- Employee information
- Procedure manuals
- FAQ management
"""

from graphmem import GraphMem, MemoryConfig, MemoryImportance
from datetime import datetime
import os

class EnterpriseKB:
    def __init__(self, org_id: str):
        self.config = MemoryConfig(
            # Use Azure OpenAI for enterprise
            llm_provider="azure_openai",
            llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
            llm_api_base=os.getenv("AZURE_ENDPOINT"),  # e.g., "https://your-resource.openai.azure.com/"
            azure_deployment="gpt-4",
            llm_model="gpt-4",
            azure_api_version="2024-02-15-preview",
            
            embedding_provider="azure_openai",
            embedding_api_key=os.getenv("AZURE_OPENAI_KEY"),
            embedding_api_base=os.getenv("AZURE_ENDPOINT"),  # Same endpoint for embeddings
            azure_embedding_deployment="text-embedding-ada-002",
            embedding_model="text-embedding-ada-002",
            
            # Enterprise Neo4j
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_username="neo4j",
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            
            # Redis for performance
            redis_url=os.getenv("REDIS_URL"),
            
            # Org isolation
            user_id=org_id,
            
            # Evolution for conflict resolution
            evolution_enabled=True,
        )
        self.memory = GraphMem(self.config)
        self.org_id = org_id
    
    def add_policy(self, policy_name: str, content: str, effective_date: str):
        """Add or update a policy document."""
        self.memory.ingest(
            f"POLICY: {policy_name} (Effective: {effective_date})\n\n{content}",
            metadata={
                "type": "policy",
                "name": policy_name,
                "effective_date": effective_date,
            },
            importance=MemoryImportance.CRITICAL,
        )
        # Evolve to supersede old policy versions
        self.memory.evolve()
    
    def add_procedure(self, name: str, steps: list[str]):
        """Add a procedure manual."""
        content = f"PROCEDURE: {name}\n\n"
        content += "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        
        self.memory.ingest(
            content,
            metadata={"type": "procedure", "name": name},
            importance=MemoryImportance.HIGH,
        )
    
    def add_employee_info(self, employee_data: dict):
        """Add employee information."""
        content = f"""
        EMPLOYEE: {employee_data['name']}
        Title: {employee_data['title']}
        Department: {employee_data['department']}
        Email: {employee_data['email']}
        Reports to: {employee_data.get('manager', 'N/A')}
        Start date: {employee_data.get('start_date', 'N/A')}
        """
        self.memory.ingest(
            content,
            metadata={"type": "employee", **employee_data},
            importance=MemoryImportance.MEDIUM,
        )
    
    def answer_question(self, question: str, department: str = None) -> dict:
        """Answer an employee's question."""
        # Add department context if provided
        if department:
            question = f"[{department} department] {question}"
        
        response = self.memory.query(question)
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [n.name for n in response.nodes[:3]],
            "needs_escalation": response.confidence < 0.5,
        }
    
    def bulk_import(self, documents: list[dict]):
        """Bulk import documents from various sources."""
        self.memory.ingest_batch(
            documents,
            max_workers=20,
            aggressive=True,
            show_progress=True,
        )
        self.memory.evolve()

# Usage
kb = EnterpriseKB("acme_corp")

# Add policies
kb.add_policy(
    "Remote Work Policy",
    "Employees may work remotely up to 3 days per week...",
    "2024-01-01"
)

# Update policy (old version auto-superseded)
kb.add_policy(
    "Remote Work Policy", 
    "Employees may work fully remote with manager approval...",
    "2024-06-01"
)

# Add procedures
kb.add_procedure("Expense Reimbursement", [
    "Submit expense report within 30 days",
    "Include receipts for items over $25",
    "Manager approval required for items over $500",
    "Finance processes within 5 business days",
])

# Answer questions
result = kb.answer_question("Can I work from home?")
print(result["answer"])
# → "Yes, employees may work fully remote with manager approval (policy effective June 2024)"
```

---

## Conclusion

GraphMem provides the memory infrastructure that transforms simple LLM wrappers into true AI agents. By handling:

- ✅ **Knowledge extraction** - Automatic entity and relationship extraction
- ✅ **Semantic storage** - Graph-based knowledge representation
- ✅ **Intelligent retrieval** - Multi-hop reasoning and alias awareness
- ✅ **Memory evolution** - Self-improving through consolidation and decay
- ✅ **Temporal validity** - Track when facts were true
- ✅ **Conflict resolution** - Automatically prefer newer information
- ✅ **Production scaling** - Redis caching, Neo4j clustering

...you can focus on building the agent logic while GraphMem handles the memory.

---

## Resources

- **GitHub**: https://github.com/Al-aminI/GraphMem
- **PyPI**: https://pypi.org/project/agentic-graph-mem/
- **Issues**: https://github.com/Al-aminI/GraphMem/issues

---

*Built with ❤️ for the AI Agent community*


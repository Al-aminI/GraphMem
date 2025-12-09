# Building AI Agents with GraphMem

> **The definitive guide to building production AI agents with human-like memory**

GraphMem provides the memory layer that transforms stateless LLMs into intelligent agents capable of learning, remembering, and evolving over time.

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

### Your First Memory-Enabled Agent

```python
from graphmem import GraphMem, MemoryConfig

# Initialize with your LLM provider
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    turso_db_path="agent_memory.db",
    evolution_enabled=True,
    auto_evolve=True,
)

memory = GraphMem(config, memory_id="my_agent", user_id="default")

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

## The Three Pillars: Ingest → Query → Evolve

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
```

---

## Agent Patterns

GraphMem enables several powerful agent patterns:

<div class="grid cards" markdown>

-   :material-chat:{ .lg .middle } **Conversational Agent**

    ---

    Persistent memory across sessions

    [:octicons-arrow-right-24: Learn more](conversational.md)

-   :material-book-open:{ .lg .middle } **Research Agent**

    ---

    Multi-document learning and synthesis

    [:octicons-arrow-right-24: Learn more](research.md)

-   :material-domain:{ .lg .middle } **Enterprise Agent**

    ---

    Multi-tenant with conflict resolution

    [:octicons-arrow-right-24: Learn more](enterprise.md)

</div>

---

## Key Features for Agents

### 1. Exhaustive Knowledge Extraction

GraphMem extracts **everything** from your text:

```python
memory.ingest("""
    In Q3 2024, Nvidia (NVDA) reported $35.1B revenue, up 94% YoY.
    CEO Jensen Huang announced Blackwell B200 shipping in early 2025.
""")

# GraphMem extracts:
# ENTITIES: Nvidia, Jensen Huang, $35.1B, 94%, Blackwell B200, Q3 2024
# RELATIONSHIPS: Jensen Huang --[is CEO of]--> Nvidia
# TEMPORAL: Revenue valid for Q3 2024, B200 ships in early 2025
```

### 2. Alias-Aware Retrieval

```python
memory.ingest("""
    Dr. Alexander Chen, also known as "The Quantum Pioneer",
    founded Quantum AI Labs. Alex Chen received his PhD from MIT.
""")

# All of these work:
memory.query("What did Dr. Chen do?")
memory.query("Who is Alexander Chen?")
memory.query("Tell me about The Quantum Pioneer")
# All return the same person's information!
```

### 3. Multi-Hop Reasoning

```python
memory.ingest("Apple was founded by Steve Jobs.")
memory.ingest("Steve Jobs also founded NeXT.")
memory.ingest("NeXT was acquired by Apple in 1997.")

response = memory.query("What's the connection between NeXT and Apple?")
# GraphMem traverses: NeXT → Steve Jobs → Apple
```

### 4. Temporal Queries

```python
memory.ingest("John was CEO from 2010-2018. Jane became CEO in 2018.")

memory.query("Who was CEO in 2015?")  # → John
memory.query("Who is CEO now?")        # → Jane
```

### 5. Conflict Resolution

```python
memory.ingest("The company has 1,000 employees.")
memory.ingest("UPDATE: The company now has 2,500 employees.")

memory.evolve()  # Resolves conflict

memory.query("How many employees?")  # → 2,500
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
    - CEO Jane Smith attributed growth to AI products
""")
```

### 2. Use Batch Ingestion

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
# After batch ingestion
memory.ingest_batch(docs)
memory.evolve()

# Or auto-evolve
config = MemoryConfig(..., auto_evolve=True)
```

### 4. Set Appropriate Importance

```python
from graphmem import MemoryImportance

# Critical (never decays)
memory.ingest(
    "Customer allergies: peanuts, shellfish",
    importance=MemoryImportance.CRITICAL,
)

# Low (may decay)
memory.ingest(
    "Customer mentioned they like coffee",
    importance=MemoryImportance.LOW,
)
```

---

## Next Steps

- [Conversational Agents](conversational.md) - Build chatbots with memory
- [Research Agents](research.md) - Multi-document synthesis
- [Enterprise Agents](enterprise.md) - Multi-tenant production systems
- [Best Practices](best-practices.md) - Production tips


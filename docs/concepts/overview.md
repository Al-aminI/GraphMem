# Core Concepts Overview

GraphMem is built on four fundamental concepts that mirror how human memory works.

## The Three Pillars

```
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│  ingest()   │        │   query()   │        │  evolve()   │
│  Learn new  │        │  Recall +   │        │  Mature     │
│  knowledge  │        │  Reasoning  │        │  memories   │
└──────┬──────┘        └──────┬──────┘        └──────┬──────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                           │
│                                                              │
│   Entities ──[relationships]──▶ Entities                    │
│                                                              │
│   With: Temporal validity, PageRank, Multi-tenant           │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Knowledge Graph

When you ingest text, GraphMem extracts:

- **Entities**: People, companies, concepts, dates, numbers
- **Relationships**: How entities connect
- **Attributes**: Properties of entities

```python
memory.ingest("Elon Musk is CEO of Tesla since 2008.")
```

Creates:

```
┌────────────┐         ┌────────────┐
│ Elon Musk  │──CEO──▶│   Tesla    │
│ (Person)   │  since  │ (Company)  │
│            │  2008   │            │
└────────────┘         └────────────┘
```

[:octicons-arrow-right-24: Learn more about Knowledge Graph](knowledge-graph.md)

---

## 2. Memory Evolution

Like human memory, GraphMem's memory evolves:

| Mechanism | What Happens | Human Equivalent |
|-----------|--------------|------------------|
| **Decay** | Unused memories fade | Forgetting curve |
| **Consolidation** | Similar memories merge | Sleep consolidation |
| **Rehydration** | Conflicts resolved | Memory updating |
| **Importance** | Hub entities prioritized | Synaptic strengthening |

```python
memory.evolve()  # Trigger evolution
```

[:octicons-arrow-right-24: Learn more about Evolution](evolution.md)

---

## 3. Temporal Validity

Every relationship has a time window:

```python
memory.ingest("John was CEO from 2010-2018")
memory.ingest("Jane became CEO in 2018")

memory.query("Who was CEO in 2015?")  # → John
memory.query("Who is CEO now?")        # → Jane
```

[:octicons-arrow-right-24: Learn more about Temporal Validity](temporal.md)

---

## 4. Multi-Tenancy

Complete data isolation between users:

```python
alice = GraphMem(config, user_id="alice")
bob = GraphMem(config, user_id="bob")

alice.ingest("I work at Google")
bob.query("Where does Alice work?")  # → "No information found"
```

[:octicons-arrow-right-24: Learn more about Multi-Tenancy](multi-tenancy.md)

---

## How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│                        GraphMem                                  │
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

## Data Flow

### Ingest Flow

1. Text is chunked semantically
2. LLM extracts entities and relationships
3. Entities are resolved (deduplicated)
4. Embeddings are generated
5. Data is stored in the graph

### Query Flow

1. Query is embedded
2. Similar entities are found
3. Related entities are traversed
4. Context is assembled
5. LLM generates answer

### Evolve Flow

1. PageRank is recalculated
2. Unused memories decay
3. Similar entities consolidate
4. Conflicts are resolved
5. Cache is invalidated


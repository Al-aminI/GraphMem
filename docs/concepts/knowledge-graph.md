# Knowledge Graph

GraphMem automatically builds a knowledge graph from your text.

## What Gets Extracted

### Entities

| Type | Examples |
|------|----------|
| Person | Elon Musk, Dr. Smith, CEO |
| Organization | Tesla, Google, Anthropic |
| Product | GPT-4, Model S, iPhone |
| Location | Austin, California, HQ |
| Date/Time | 2024, Q3, January |
| Number | $35B, 26,000, 94% |
| Concept | AI, sustainability, revenue |

### Relationships

| Relationship | Example |
|--------------|---------|
| CEO_OF | Elon Musk → Tesla |
| FOUNDED | Steve Jobs → Apple |
| WORKS_AT | Alice → Google |
| LOCATED_IN | Tesla HQ → Austin |
| ACQUIRED | Apple → NeXT |
| CREATED | Anthropic → Claude |

### Attributes

| Attribute | Example |
|-----------|---------|
| Aliases | "Elon", "Musk", "CEO" |
| Description | "Electric vehicle company" |
| Properties | revenue, employees, founded |

---

## Example Extraction

### Input Text

```python
memory.ingest("""
    In Q3 2024, Nvidia (NVDA) reported $35.1B revenue, up 94% YoY.
    CEO Jensen Huang announced Blackwell B200 shipping in early 2025.
    The company has 26,000 employees worldwide.
""")
```

### Extracted Graph

```
                    ┌─────────────┐
                    │   Nvidia    │
                    │ (Company)   │
                    │ aliases:    │
                    │ NVDA        │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Jensen Huang │    │   $35.1B    │    │Blackwell    │
│  (Person)   │    │  (Revenue)  │    │   B200      │
│             │    │  Q3 2024    │    │ (Product)   │
│ CEO_OF ─────┼────┼─────────────│    │ ships: 2025 │
└─────────────┘    │  +94% YoY   │    └─────────────┘
                   └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   26,000    │
                   │ (Employees) │
                   └─────────────┘
```

---

## Alias Resolution

GraphMem automatically handles entity aliases:

```python
memory.ingest("""
    Dr. Alexander Chen, also known as "The Quantum Pioneer",
    founded Quantum AI Labs. Alex Chen received his PhD from MIT.
""")
```

All these refer to the same entity:
- Dr. Alexander Chen
- The Quantum Pioneer
- Alex Chen

```python
# All queries work:
memory.query("What did Dr. Chen do?")
memory.query("Who is Alexander Chen?")
memory.query("Tell me about The Quantum Pioneer")
memory.query("What did Alex Chen study?")
```

---

## Entity Resolution

When ingesting multiple documents, GraphMem resolves duplicates:

```python
memory.ingest("Elon Musk is CEO of Tesla")
memory.ingest("Tesla's CEO Musk announced new plans")
memory.ingest("Mr. Musk spoke at the conference")
```

Result: **One** "Elon Musk" entity with all relationships, not three duplicates.

### Resolution Strategy

1. **Exact match**: Same name
2. **Alias match**: Known alias
3. **Embedding similarity**: Similar meaning
4. **LLM verification**: Final check for edge cases

---

## Graph Traversal

GraphMem can traverse relationships to answer complex questions:

```python
memory.ingest("Apple was founded by Steve Jobs.")
memory.ingest("Steve Jobs also founded NeXT.")
memory.ingest("NeXT was acquired by Apple in 1997.")
memory.ingest("Tim Cook worked at Compaq before Apple.")

response = memory.query("What's the connection between NeXT and Tim Cook?")
```

Traversal path:
```
NeXT → Steve Jobs → Apple → Tim Cook
```

Answer: *"NeXT was founded by Steve Jobs and acquired by Apple in 1997. Tim Cook currently works at Apple as CEO."*

---

## Community Detection

Related entities are grouped into communities:

```
┌─────────────────────────────────────────┐
│           TECH LEADERSHIP               │
│                                          │
│  Elon Musk ─── Tesla ─── SpaceX         │
│      │                                   │
│  Jensen Huang ─── Nvidia                │
│      │                                   │
│  Tim Cook ─── Apple                     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           AI COMPANIES                   │
│                                          │
│  OpenAI ─── GPT-4 ─── Sam Altman        │
│      │                                   │
│  Anthropic ─── Claude ─── Dario Amodei  │
└─────────────────────────────────────────┘
```

Communities are used for:
- High-level summaries
- Cross-cluster queries
- Topic identification

---

## PageRank Importance

Hub entities (connected to many things) score higher:

```python
# PageRank scores
Elon Musk:      PR = 1.000  ████████████████████  # Hub
Tesla:          PR = 0.774  ███████████████
SpaceX:         PR = 0.774  ███████████████
Austin:         PR = 0.520  ██████████            # Peripheral
```

Higher PageRank = More likely to be retrieved and weighted in answers.


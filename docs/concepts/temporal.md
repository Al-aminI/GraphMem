# Temporal Validity

GraphMem tracks when facts are true, enabling point-in-time queries.

## The Problem

Traditional RAG systems can't answer:
- "Who was CEO in 2015?"
- "What was our policy before the update?"
- "What contracts were active last quarter?"

They only know the current state.

## The Solution

Every relationship in GraphMem has a time interval:

```python
@dataclass
class MemoryEdge:
    valid_from: datetime   # When the fact became true
    valid_until: datetime  # When it stopped being true (None = current)
```

---

## How It Works

### Automatic Extraction

GraphMem extracts temporal information from text:

```python
memory.ingest("John Smith was CEO of ACME from 2010 to 2018")
memory.ingest("Jane Doe became CEO of ACME in July 2018")
```

Creates:

```
John ──CEO_OF──▶ ACME   [2010-01-01 to 2018-06-30]
Jane ──CEO_OF──▶ ACME   [2018-07-01 to NULL]       ← Current
```

### Point-in-Time Queries

```python
memory.query("Who was CEO in 2015?")      # → "John Smith"
memory.query("Who is CEO now?")           # → "Jane Doe"
memory.query("Who was CEO in 2019?")      # → "Jane Doe"
```

---

## Timeline Visualization

```
Timeline:  2010      2015      2018      2020      2025
           │         │         │         │         │
           ▼         ▼         ▼         ▼         ▼

John ──CEO_OF──▶ ACME  ════════════════╗
[2010-01-01 to 2018-06-30]              ║
                                        ║
Jane ──CEO_OF──▶ ACME                   ╚════════════════════════
[2018-07-01 to NULL]                     ← Still current
```

---

## API Usage

### Check Validity at a Point in Time

```python
from datetime import datetime

edge = memory.get_edge("john_ceo_acme")

# Check if valid in 2015
edge.is_valid_at(datetime(2015, 6, 1))  # True

# Check if valid now
edge.is_valid_at(datetime.utcnow())     # False
```

### Mark Relationship as Ended

```python
# When John leaves
edge.supersede(end_time=datetime(2018, 6, 30))
```

### Query Edges at Specific Time

```python
# Get all CEO relationships as of 2015
edges = store.query_edges_at_time(
    memory_id="company_kb",
    query_time=datetime(2015, 6, 1),
    relation_type="CEO_OF"
)
```

---

## Use Cases

### Corporate Governance

```python
memory.ingest("""
    Sarah Johnson joined as CFO in 2019.
    She was promoted to CEO in January 2023.
    Michael Chen took over as CFO when Sarah was promoted.
""")

memory.query("Who was CFO in 2021?")  # → Sarah Johnson
memory.query("Who is CFO now?")       # → Michael Chen
```

### Policy Tracking

```python
memory.ingest("Our return policy was 30 days until 2023.")
memory.ingest("UPDATE: As of June 2023, returns accepted for 60 days.")

memory.query("What was the return policy in 2022?")  # → 30 days
memory.query("What is the current return policy?")   # → 60 days
```

### Employment History

```python
memory.ingest("Alice worked at Google from 2018-2021.")
memory.ingest("Alice joined Meta in 2021 and is still there.")

memory.query("Where did Alice work in 2019?")  # → Google
memory.query("Where does Alice work now?")     # → Meta
```

### Contract Management

```python
memory.ingest("""
    Contract #123 with Acme Corp:
    - Effective: January 1, 2023
    - Expires: December 31, 2025
    - Value: $1.2M annually
""")

memory.query("Is contract 123 active today?")  # → Yes
memory.query("What contracts expire in 2025?")  # → Contract #123
```

---

## Conflict Resolution

When facts conflict, temporal validity helps resolve them:

```python
# Old information
memory.ingest("The company has 1,000 employees.")

# New information (supersedes old)
memory.ingest("As of Q4 2024, the company has 2,500 employees.")

memory.evolve()  # Resolves conflict

memory.query("How many employees?")  # → 2,500 (current)
memory.query("How many employees in early 2024?")  # → 1,000
```

---

## Storage Implementation

### Neo4j

```cypher
CREATE (john:Entity {name: 'John Smith', type: 'Person'})
CREATE (acme:Entity {name: 'ACME Corp', type: 'Company'})
CREATE (john)-[:RELATIONSHIP {
    relation_type: 'CEO_OF',
    valid_from: datetime('2010-01-01'),
    valid_until: datetime('2018-06-30')
}]->(acme)
```

### Turso (SQLite)

```sql
CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    target_id TEXT,
    relation_type TEXT,
    valid_from DATETIME,
    valid_until DATETIME
);

-- Query for a specific time
SELECT * FROM edges 
WHERE valid_from <= '2015-06-01' 
  AND (valid_until IS NULL OR valid_until >= '2015-06-01');
```

---

## Best Practices

1. **Include dates in your text** - GraphMem extracts them automatically
2. **Use clear temporal language** - "from X to Y", "since", "until", "as of"
3. **Mark updates explicitly** - "UPDATE:", "NEW:", "AS OF [date]"
4. **Evolve regularly** - To resolve temporal conflicts


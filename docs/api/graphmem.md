# GraphMem Class

The main class for interacting with GraphMem.

## Constructor

```python
from graphmem import GraphMem, MemoryConfig

memory = GraphMem(
    config: MemoryConfig,
    user_id: str = "default",
    memory_id: str = "default",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | MemoryConfig | Required | Configuration object |
| `user_id` | str | "default" | User ID for multi-tenancy |
| `memory_id` | str | "default" | Memory context ID |

---

## Methods

### ingest()

Ingest content into memory.

```python
result = memory.ingest(
    content: str,
    metadata: dict = None,
    importance: MemoryImportance = MemoryImportance.MEDIUM,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | str | Text content to ingest |
| `metadata` | dict | Optional metadata |
| `importance` | MemoryImportance | Importance level |

**Returns:** `dict`

```python
{
    "entities": int,        # Number of entities extracted
    "relationships": int,   # Number of relationships
    "chunks": int,          # Number of chunks processed
}
```

**Example:**

```python
result = memory.ingest(
    "Tesla is led by CEO Elon Musk. Founded in 2003.",
    metadata={"source": "news"},
    importance=MemoryImportance.HIGH,
)
print(f"Extracted {result['entities']} entities")
```

---

### ingest_batch()

Ingest multiple documents in parallel.

```python
result = memory.ingest_batch(
    documents: list[dict],
    max_workers: int = 10,
    aggressive: bool = False,
    show_progress: bool = False,
    rebuild_communities: bool = True,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `documents` | list[dict] | List of {"content": str, ...} |
| `max_workers` | int | Parallel workers |
| `aggressive` | bool | Use aggressive rate limiting |
| `show_progress` | bool | Show progress bar |
| `rebuild_communities` | bool | Rebuild communities after |

**Returns:** `dict`

```python
{
    "documents_processed": int,
    "documents_failed": int,
    "documents_skipped": int,
    "total_entities": int,
    "total_relationships": int,
    "elapsed_seconds": float,
}
```

---

### query()

Query the memory.

```python
response = memory.query(
    question: str,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `question` | str | Question to answer |

**Returns:** `QueryResponse`

```python
@dataclass
class QueryResponse:
    answer: str              # Generated answer
    confidence: float        # Confidence score (0-1)
    context: str             # Context used for answer
    nodes: list[MemoryNode]  # Retrieved entities
    edges: list[MemoryEdge]  # Retrieved relationships
    latency_ms: float        # Query latency
```

**Example:**

```python
response = memory.query("Who is the CEO of Tesla?")
print(response.answer)       # "Elon Musk"
print(response.confidence)   # 0.95
print(len(response.nodes))   # 3
```

---

### evolve()

Trigger memory evolution.

```python
events = memory.evolve(
    force: bool = False,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `force` | bool | Force evolution even if recent |

**Returns:** `list[EvolutionEvent]`

```python
@dataclass
class EvolutionEvent:
    evolution_type: str   # CONSOLIDATION, DECAY, etc.
    description: str      # Human-readable description
    entities_affected: int
```

**Example:**

```python
events = memory.evolve()
for event in events:
    print(f"{event.evolution_type}: {event.description}")
```

---

### get_stats()

Get memory statistics.

```python
stats = memory.get_stats()
```

**Returns:** `dict`

```python
{
    "total_entities": int,
    "total_relationships": int,
    "total_communities": int,
    "unique_entity_types": list[str],
}
```

---

### clear()

Clear all memory data.

```python
memory.clear()
```

!!! warning "Destructive"
    This permanently deletes all data for the current user_id/memory_id.

---

## Properties

### store

Access the underlying storage backend.

```python
store = memory.store  # Neo4jStore, TursoStore, or InMemoryStore
```

### cache

Access the cache layer (if configured).

```python
cache = memory.cache  # RedisCache or None
```

---

## Full Example

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance

# Configure
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    turso_db_path="memory.db",
    evolution_enabled=True,
)

# Initialize
memory = GraphMem(config, user_id="alice", memory_id="chat")

# Ingest
memory.ingest("Alice works at Google as a senior engineer.")
memory.ingest("Alice is interested in machine learning.")

# Query
response = memory.query("What does Alice do?")
print(response.answer)

# Evolve
events = memory.evolve()

# Stats
stats = memory.get_stats()
print(f"Entities: {stats['total_entities']}")
```


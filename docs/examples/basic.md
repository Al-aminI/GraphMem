# Basic Usage Examples

Simple examples to get started with GraphMem.

## Hello World

```python
from graphmem import GraphMem, MemoryConfig

# Configure
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
)

# Initialize
memory = GraphMem(config)

# Learn
memory.ingest("Tesla is led by CEO Elon Musk. Founded in 2003.")

# Recall
response = memory.query("Who is the CEO of Tesla?")
print(response.answer)  # "Elon Musk"
```

## Multiple Documents

```python
# Ingest multiple facts
memory.ingest("Tesla is led by CEO Elon Musk.")
memory.ingest("SpaceX was founded by Elon Musk in 2002.")
memory.ingest("Neuralink develops brain-computer interfaces.")

# Query across documents
response = memory.query("What companies does Elon Musk lead?")
print(response.answer)  # "Tesla, SpaceX, and Neuralink"
```

## With Persistence

```python
from graphmem import GraphMem, MemoryConfig

# Configure with Turso for persistence
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    turso_db_path="my_memory.db",
)

memory = GraphMem(config)

# Data persists between restarts!
memory.ingest("Important information...")
```

## With Evolution

```python
# Ingest data
memory.ingest("The company has 1000 employees.")
memory.ingest("The company now has 2500 employees.")

# Evolve to resolve conflicts
events = memory.evolve()
for event in events:
    print(f"{event.evolution_type}: {event.description}")

# Query returns updated info
response = memory.query("How many employees?")
print(response.answer)  # "2500"
```

## Setting Importance

```python
from graphmem import MemoryImportance

# Critical (never decays)
memory.ingest(
    "Customer allergies: peanuts",
    importance=MemoryImportance.CRITICAL,
)

# Low (may decay over time)
memory.ingest(
    "Customer likes coffee",
    importance=MemoryImportance.LOW,
)
```

## Batch Ingestion

```python
documents = [
    {"id": "1", "content": "Document 1 content..."},
    {"id": "2", "content": "Document 2 content..."},
    {"id": "3", "content": "Document 3 content..."},
]

result = memory.ingest_batch(
    documents,
    max_workers=10,
    aggressive=True,
)

print(f"Processed: {result['documents_processed']}")
print(f"Entities: {result['total_entities']}")
```

## Getting Statistics

```python
stats = memory.get_stats()
print(f"Entities: {stats['total_entities']}")
print(f"Relationships: {stats['total_relationships']}")
print(f"Communities: {stats['total_communities']}")
```


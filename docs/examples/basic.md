# Basic Usage Examples

Simple examples to get started with GraphMem.

!!! warning "Enable Persistence!"
    By default, GraphMem uses **in-memory storage**. Add `turso_db_path` to persist your data between restarts!

## Hello World (with Persistence)

```python
from graphmem import GraphMem, MemoryConfig

# Configure with OpenAI + Persistence
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ IMPORTANT: Enable persistence!
    turso_db_path="my_agent_memory.db",
)

# Initialize with a memory_id to find your data again
memory = GraphMem(config, memory_id="my_agent")

# Learn
memory.ingest("Tesla is led by CEO Elon Musk. Founded in 2003.")

# Recall
response = memory.query("Who is the CEO of Tesla?")
print(response.answer)  # "Elon Musk"

# Data survives restarts!
```

## Using Custom Base URLs

For OpenRouter, Together, Groq, or local models:

```python
# OpenRouter
config = MemoryConfig(
    llm_provider="openai_compatible",
    llm_api_key="sk-or-v1-...",
    llm_api_base="https://openrouter.ai/api/v1",
    llm_model="google/gemini-2.0-flash-001",
    embedding_provider="openai_compatible",
    embedding_api_key="sk-or-v1-...",
    embedding_api_base="https://openrouter.ai/api/v1",
    embedding_model="openai/text-embedding-3-small",
    turso_db_path="my_agent_memory.db",  # ✅ Don't forget persistence!
)

# Local Ollama
config = MemoryConfig(
    llm_provider="openai_compatible",
    llm_api_key="not-needed",
    llm_api_base="http://localhost:11434/v1",
    llm_model="llama3.2",
    embedding_provider="openai_compatible",
    embedding_api_key="not-needed",
    embedding_api_base="http://localhost:11434/v1",
    embedding_model="nomic-embed-text",
    turso_db_path="my_agent_memory.db",  # ✅ Don't forget persistence!
)

# Azure OpenAI
config = MemoryConfig(
    llm_provider="azure_openai",
    llm_api_key="your-azure-key",
    llm_api_base="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4",
    llm_model="gpt-4",
    azure_api_version="2024-02-15-preview",
    embedding_provider="azure_openai",
    embedding_api_key="your-azure-key",
    embedding_api_base="https://your-resource.openai.azure.com/",
    azure_embedding_deployment="text-embedding-ada-002",
    embedding_model="text-embedding-ada-002",
    turso_db_path="my_agent_memory.db",  # ✅ Don't forget persistence!
)
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

## With Persistence + Cloud Sync

```python
from graphmem import GraphMem, MemoryConfig

# Local persistence only
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ REQUIRED for persistence
    turso_db_path="my_memory.db",
)

# Local + Cloud sync (for backup & multi-device)
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ Local file (REQUIRED)
    turso_db_path="my_memory.db",
    
    # ✅ Cloud sync (OPTIONAL - for backup)
    turso_url="libsql://your-db.turso.io",
    turso_auth_token="your-token",
)

# Always use a consistent memory_id!
memory = GraphMem(config, memory_id="my_agent")

# Data persists between restarts AND syncs to cloud!
memory.ingest("Important information...")
```

!!! tip "How Turso Persistence Works"
    - `turso_db_path` creates a **local SQLite file** (~1ms reads)
    - `turso_url` + `turso_auth_token` enable **automatic cloud sync**
    - Data is always read/written locally first (fast, works offline)
    - Cloud sync happens automatically in the background

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


# Quick Start

Get GraphMem running in 5 minutes.

## Step 1: Install

```bash
pip install agentic-graph-mem
```

For persistent storage (recommended):
```bash
pip install "agentic-graph-mem[libsql]"
```

## Step 2: Configure

!!! warning "Important: Enable Persistence!"
    By default, GraphMem uses **in-memory storage** (data is lost on restart). Add `turso_db_path` to persist your data!

=== "With Persistence (Recommended)"

    ```python
    from graphmem import GraphMem, MemoryConfig

    config = MemoryConfig(
        # LLM for extraction and querying
        llm_provider="openai",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        
        # Embeddings for semantic search
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
        
        # ✅ PERSISTENCE: Data survives restarts!
        turso_db_path="my_agent_memory.db",
    )

    memory = GraphMem(config)
    ```

=== "In-Memory Only (Dev/Testing)"

    ```python
    from graphmem import GraphMem, MemoryConfig

    config = MemoryConfig(
        # LLM for extraction and querying
        llm_provider="openai",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        
        # Embeddings for semantic search
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
        # ⚠️ No turso_db_path = data lost on restart!
    )

    memory = GraphMem(config)
    ```

## Step 3: Learn (Ingest)

```python
# Feed information to memory
memory.ingest("""
    Tesla, Inc. is an American electric vehicle company.
    Elon Musk is the CEO. Founded in 2003, Tesla's mission
    is to accelerate the transition to sustainable energy.
""")

memory.ingest("""
    SpaceX is led by Elon Musk as CEO. Founded in 2002,
    SpaceX designs rockets. Goal: make humanity multiplanetary.
""")
```

## Step 4: Recall (Query)

```python
# Ask questions
response = memory.query("Who is the CEO of Tesla?")
print(response.answer)  # "Elon Musk"

response = memory.query("What companies does Elon Musk lead?")
print(response.answer)  # "Tesla and SpaceX"
```

## Step 5: Evolve

```python
# Let memory mature (consolidate, decay, improve)
memory.evolve()
```

---

## Complete Example

```python
from graphmem import GraphMem, MemoryConfig

# Configure with persistence
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ Enable persistence (REQUIRED for production!)
    turso_db_path="my_agent_memory.db",
)

# Initialize with a consistent memory_id to find your data again
memory = GraphMem(config, memory_id="my_agent")

# Learn
memory.ingest("Tesla is led by CEO Elon Musk. Founded in 2003.")
memory.ingest("SpaceX, founded by Elon Musk in 2002, builds rockets.")
memory.ingest("Neuralink develops brain-computer interfaces.")

# Recall
response = memory.query("What companies does Elon Musk lead?")
print(response.answer)
# → "Elon Musk leads Tesla, SpaceX, and Neuralink."

print(f"Confidence: {response.confidence}")
# → 0.95

# Evolve
events = memory.evolve()
for event in events:
    print(f"{event.evolution_type}: {event.description}")

# Close when done (ensures data is synced)
memory.close()
```

!!! tip "Use `memory_id` for Consistent Access"
    Always provide a `memory_id` when creating `GraphMem` to find your data again:
    ```python
    # First session
    memory = GraphMem(config, memory_id="my_agent")
    memory.ingest("...")
    
    # Later session - same memory_id loads your data!
    memory = GraphMem(config, memory_id="my_agent")
    response = memory.query("...")  # Finds your previously ingested data
    ```

---

## Using Alternative Providers

=== "Azure OpenAI"

    ```python
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
        
        # ✅ Don't forget persistence!
        turso_db_path="my_agent_memory.db",
    )
    ```

=== "OpenRouter"

    ```python
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="sk-or-v1-...",
        llm_api_base="https://openrouter.ai/api/v1",
        llm_model="google/gemini-2.0-flash-001",
        
        embedding_provider="openai_compatible",
        embedding_api_key="sk-or-v1-...",
        embedding_api_base="https://openrouter.ai/api/v1",
        embedding_model="openai/text-embedding-3-small",
        
        # ✅ Don't forget persistence!
        turso_db_path="my_agent_memory.db",
    )
    ```

=== "Local (Ollama)"

    ```python
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="not-needed",
        llm_api_base="http://localhost:11434/v1",
        llm_model="llama3.2",
        
        embedding_provider="openai_compatible",
        embedding_api_key="not-needed",
        embedding_api_base="http://localhost:11434/v1",
        embedding_model="nomic-embed-text",
        
        # ✅ Don't forget persistence!
        turso_db_path="my_agent_memory.db",
    )
    ```

=== "Anthropic"

    ```python
    config = MemoryConfig(
        llm_provider="anthropic",
        llm_api_key="sk-ant-...",
        llm_model="claude-3-5-sonnet-20241022",
        
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
        
        # ✅ Don't forget persistence!
        turso_db_path="my_agent_memory.db",
    )
    ```

---

## Adding Cloud Sync (Optional)

For backup and multi-device sync, add Turso Cloud credentials:

```python
config = MemoryConfig(
    # ... LLM config ...
    
    # Local persistence (REQUIRED)
    turso_db_path="my_agent_memory.db",
    
    # Cloud sync (OPTIONAL - for backup & multi-device)
    turso_url="libsql://your-db.turso.io",
    turso_auth_token="your-token",
)
```

!!! info "How it works"
    - `turso_db_path` creates a local SQLite file (always fast, works offline)
    - `turso_url` + `turso_auth_token` enable automatic cloud sync
    - Data is read/written locally first, then synced to cloud

---

## Next Steps

- [Configuration](configuration.md) - Full configuration options
- [Storage Backends](storage.md) - Choose the right storage
- [Building Agents](../agents/guide.md) - Build production agents


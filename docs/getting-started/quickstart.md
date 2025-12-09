# Quick Start

Get GraphMem running in 5 minutes.

## Step 1: Install

```bash
pip install agentic-graph-mem
```

## Step 2: Configure

```python
from graphmem import GraphMem, MemoryConfig

config = MemoryConfig(
    # LLM for extraction and querying
    llm_provider="openai",
    llm_api_key="sk-...",  # Your OpenAI API key
    llm_model="gpt-4o-mini",
    
    # Embeddings for semantic search
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
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
    )
    ```

=== "OpenRouter"

    ```python
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="sk-or-v1-...",
        llm_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        llm_model="google/gemini-2.0-flash-001",
        
        embedding_provider="openai_compatible",
        embedding_api_key="sk-or-v1-...",
        embedding_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        embedding_model="openai/text-embedding-3-small",
    )
    ```

=== "Local (Ollama)"

    ```python
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="not-needed",
        llm_api_base="http://localhost:11434/v1",  # Ollama base URL
        llm_model="llama3.2",
        
        embedding_provider="openai_compatible",
        embedding_api_key="not-needed",
        embedding_api_base="http://localhost:11434/v1",  # Ollama base URL
        embedding_model="nomic-embed-text",
    )
    ```

=== "Anthropic"

    ```python
    config = MemoryConfig(
        llm_provider="anthropic",
        llm_api_key="sk-ant-...",
        llm_model="claude-3-5-sonnet-20241022",
        
        # Still need OpenAI for embeddings
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
    )
    ```

---

## Adding Persistence

By default, GraphMem uses in-memory storage. Add persistence with Turso:

```python
config = MemoryConfig(
    # ... LLM config ...
    
    # Add persistence
    turso_db_path="my_agent_memory.db",
)

memory = GraphMem(config)

# Data now persists between restarts!
memory.ingest("Important information...")
```

---

## Next Steps

- [Configuration](configuration.md) - Full configuration options
- [Storage Backends](storage.md) - Choose the right storage
- [Building Agents](../agents/guide.md) - Build production agents


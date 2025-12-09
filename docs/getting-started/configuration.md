# Configuration

GraphMem is configured using the `MemoryConfig` class.

## Basic Configuration

```python
from graphmem import MemoryConfig

config = MemoryConfig(
    # Required: LLM provider
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    
    # Required: Embedding provider
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
)
```

## LLM Configuration

### OpenAI

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",  # or gpt-4o, gpt-4-turbo
)
```

### Azure OpenAI

```python
config = MemoryConfig(
    llm_provider="azure_openai",
    llm_api_key="your-azure-key",
    llm_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
    azure_deployment="gpt-4",
    llm_model="gpt-4",
    azure_api_version="2024-02-15-preview",  # Optional
)
```

### OpenAI-Compatible (OpenRouter, Together, Groq, etc.)

```python
config = MemoryConfig(
    llm_provider="openai_compatible",
    llm_api_key="your-key",
    llm_api_base="https://openrouter.ai/api/v1",  # Custom base URL
    llm_model="google/gemini-2.0-flash-001",
    
    embedding_provider="openai_compatible",
    embedding_api_key="your-key",
    embedding_api_base="https://openrouter.ai/api/v1",  # Custom base URL
    embedding_model="openai/text-embedding-3-small",
)
```

### Local LLMs (Ollama, vLLM, LM Studio)

```python
config = MemoryConfig(
    llm_provider="openai_compatible",
    llm_api_key="not-needed",  # Local models often don't need keys
    llm_api_base="http://localhost:11434/v1",  # Ollama
    llm_model="llama3.2",
    
    embedding_provider="openai_compatible",
    embedding_api_key="not-needed",
    embedding_api_base="http://localhost:11434/v1",
    embedding_model="nomic-embed-text",
)
```

### Anthropic

```python
config = MemoryConfig(
    llm_provider="anthropic",
    llm_api_key="sk-ant-...",
    llm_model="claude-3-5-sonnet-20241022",
)
```

## Embedding Configuration

### OpenAI Embeddings

```python
config = MemoryConfig(
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",  # or text-embedding-3-large
)
```

### Azure OpenAI Embeddings

```python
config = MemoryConfig(
    embedding_provider="azure_openai",
    embedding_api_key="your-azure-key",
    embedding_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
    azure_embedding_deployment="text-embedding-ada-002",
    embedding_model="text-embedding-ada-002",
    azure_api_version="2024-02-15-preview",  # Optional
)
```

## Storage Configuration

### In-Memory (Default)

```python
config = MemoryConfig(
    # No storage config needed - uses in-memory by default
)
```

### Turso (SQLite Persistence)

!!! warning "Important"
    `turso_db_path` is **required** to enable Turso. Just providing `turso_url` without `turso_db_path` will NOT use Turso!

```python
# Local-only (no cloud sync)
config = MemoryConfig(
    turso_db_path="my_memory.db",  # ✅ REQUIRED - local SQLite file
)

# With cloud sync (local + cloud backup)
config = MemoryConfig(
    turso_db_path="my_memory.db",  # ✅ REQUIRED - local SQLite file
    turso_url="libsql://your-db.turso.io",  # Optional - cloud sync URL
    turso_auth_token="your-token",  # Optional - cloud auth token
)
```

Turso uses an **offline-first architecture**: all data is stored locally, with optional cloud sync for backup and multi-device access. See [Storage Backends](storage.md) for details.

### Neo4j

```python
config = MemoryConfig(
    neo4j_uri="neo4j+s://xxx.databases.neo4j.io",
    neo4j_username="neo4j",
    neo4j_password="your-password",
)
```

### Redis Cache

```python
config = MemoryConfig(
    redis_url="redis://default:password@host:port",
)
```

## Evolution Configuration

```python
config = MemoryConfig(
    # Enable memory evolution
    evolution_enabled=True,
    
    # Auto-evolve after each ingestion
    auto_evolve=True,
    
    # Memory decay settings
    decay_enabled=True,
    decay_half_life_days=30,  # How fast memories fade
    
    # Consolidation threshold (0.0 - 1.0)
    consolidation_threshold=0.85,  # Higher = less aggressive merging
)
```

## Multi-Tenant Configuration

```python
from graphmem import GraphMem

# User-specific memory
memory = GraphMem(
    config,
    user_id="alice",      # Isolate by user
    memory_id="chat_1",   # Isolate by session/context
)
```

## All Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | str | Required | LLM provider name |
| `llm_api_key` | str | Required | API key for LLM |
| `llm_model` | str | Required | Model name |
| `llm_api_base` | str | None | Custom LLM API base URL (for OpenRouter, Together, local models) |
| `embedding_provider` | str | Required | Embedding provider |
| `embedding_api_key` | str | Required | API key for embeddings |
| `embedding_model` | str | Required | Embedding model name |
| `embedding_api_base` | str | None | Custom embedding API base URL (for OpenRouter, Together, local models) |
| `azure_endpoint` | str | None | Azure OpenAI endpoint |
| `azure_deployment` | str | None | Azure LLM deployment |
| `azure_embedding_deployment` | str | None | Azure embedding deployment |
| `azure_api_version` | str | "2024-02-15-preview" | Azure API version |
| `turso_db_path` | str | None | **Required for Turso** - Local SQLite file path |
| `turso_url` | str | None | Optional - Turso cloud URL for sync (requires `turso_db_path`) |
| `turso_auth_token` | str | None | Optional - Turso auth token (requires `turso_url`) |
| `neo4j_uri` | str | None | Neo4j connection URI |
| `neo4j_username` | str | None | Neo4j username |
| `neo4j_password` | str | None | Neo4j password |
| `redis_url` | str | None | Redis connection URL |
| `evolution_enabled` | bool | True | Enable evolution |
| `auto_evolve` | bool | False | Auto-evolve on ingest |
| `decay_enabled` | bool | True | Enable memory decay |
| `decay_half_life_days` | int | 30 | Decay half-life |
| `consolidation_threshold` | float | 0.85 | Merge threshold |

## Environment Variables

You can also use environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export NEO4J_URI="neo4j+s://..."
export NEO4J_PASSWORD="..."
export REDIS_URL="redis://..."
```

```python
import os

config = MemoryConfig(
    llm_provider="openai",
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key=os.getenv("OPENAI_API_KEY"),
    embedding_model="text-embedding-3-small",
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
)
```


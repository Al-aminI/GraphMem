# MemoryConfig

Configuration class for GraphMem.

## Constructor

```python
from graphmem import MemoryConfig

config = MemoryConfig(
    # LLM Configuration
    llm_provider: str,
    llm_api_key: str,
    llm_model: str,
    llm_api_base: str = None,
    
    # Embedding Configuration
    embedding_provider: str,
    embedding_api_key: str,
    embedding_model: str,
    embedding_api_base: str = None,
    
    # Azure Configuration
    azure_endpoint: str = None,
    azure_deployment: str = None,
    azure_embedding_deployment: str = None,
    azure_api_version: str = "2024-02-15-preview",
    
    # Storage Configuration
    turso_db_path: str = None,
    turso_url: str = None,
    turso_auth_token: str = None,
    neo4j_uri: str = None,
    neo4j_username: str = "neo4j",
    neo4j_password: str = None,
    redis_url: str = None,
    
    # Evolution Configuration
    evolution_enabled: bool = True,
    auto_evolve: bool = False,
    decay_enabled: bool = True,
    decay_half_life_days: int = 30,
    consolidation_threshold: float = 0.85,
)
```

---

## Parameters

### LLM Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llm_provider` | str | Yes | Provider: "openai", "azure_openai", "anthropic", "openai_compatible", "ollama" |
| `llm_api_key` | str | Yes | API key for the provider |
| `llm_model` | str | Yes | Model name (e.g., "gpt-4o-mini") |
| `llm_api_base` | str | No | Custom API base URL (for OpenRouter, Azure, local models) |

### Embedding Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `embedding_provider` | str | Yes | Provider for embeddings |
| `embedding_api_key` | str | Yes | API key for embeddings |
| `embedding_model` | str | Yes | Model (e.g., "text-embedding-3-small") |
| `embedding_api_base` | str | No | Custom API base URL (for OpenRouter, Azure, local models) |

### Azure Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `azure_deployment` | str | For Azure | LLM deployment name |
| `azure_embedding_deployment` | str | For Azure | Embedding deployment name |
| `azure_api_version` | str | No | API version (default: "2024-02-15-preview") |

!!! note "Azure Base URL"
    For Azure OpenAI, use `llm_api_base` and `embedding_api_base` to specify your Azure endpoint (e.g., `https://your-resource.openai.azure.com/`).

### Storage Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `turso_db_path` | str | No | Local SQLite file path |
| `turso_url` | str | No | Turso cloud URL |
| `turso_auth_token` | str | No | Turso auth token |
| `neo4j_uri` | str | No | Neo4j connection URI |
| `neo4j_username` | str | No | Neo4j username |
| `neo4j_password` | str | No | Neo4j password |
| `redis_url` | str | No | Redis connection URL |

### Evolution Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evolution_enabled` | bool | True | Enable evolution |
| `auto_evolve` | bool | False | Auto-evolve on ingest |
| `decay_enabled` | bool | True | Enable memory decay |
| `decay_half_life_days` | int | 30 | Decay half-life |
| `consolidation_threshold` | float | 0.85 | Merge threshold (0-1) |

---

## Examples

### OpenAI

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
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
    azure_api_version="2024-02-15-preview",
    
    embedding_provider="azure_openai",
    embedding_api_key="your-azure-key",
    embedding_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
    azure_embedding_deployment="text-embedding-ada-002",
    embedding_model="text-embedding-ada-002",
)
```

### OpenRouter

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

### Local Models (Ollama)

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

### With Turso

```python
config = MemoryConfig(
    # LLM config...
    turso_db_path="my_memory.db",
)
```

### With Neo4j + Redis

```python
config = MemoryConfig(
    # LLM config...
    neo4j_uri="neo4j+s://xxx.databases.neo4j.io",
    neo4j_password="your-password",
    redis_url="redis://default:password@host:port",
)
```

### Full Production

```python
import os

config = MemoryConfig(
    # LLM
    llm_provider="azure_openai",
    llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
    llm_api_base=os.getenv("AZURE_ENDPOINT"),  # e.g., "https://your-resource.openai.azure.com/"
    azure_deployment="gpt-4",
    llm_model="gpt-4",
    azure_api_version="2024-02-15-preview",
    
    # Embeddings
    embedding_provider="azure_openai",
    embedding_api_key=os.getenv("AZURE_OPENAI_KEY"),
    embedding_api_base=os.getenv("AZURE_ENDPOINT"),  # Same endpoint for embeddings
    azure_embedding_deployment="text-embedding-ada-002",
    embedding_model="text-embedding-ada-002",
    
    # Storage
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    redis_url=os.getenv("REDIS_URL"),
    
    # Evolution
    evolution_enabled=True,
    auto_evolve=False,
    decay_half_life_days=30,
)
```


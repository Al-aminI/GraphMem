# Storage Backends

GraphMem supports multiple storage backends for different use cases.

## Storage Decision Tree

```
"Do you need data to persist between restarts?"
                    │
       ┌────────────┴────────────┐
       │                         │
       NO                       YES
       │                         │
       ▼                         ▼
 ┌─────────────┐    "Do you need complex graph queries?"
 │  InMemory   │                 │
 │             │    ┌────────────┴────────────┐
 │ • Dev/Test  │   NO                        YES
 │ • Quick POC │    │                         │
 └─────────────┘    ▼                         ▼
             ┌─────────────┐         ┌─────────────┐
             │   TURSO     │         │   NEO4J     │
             │             │         │             │
             │ • Offline   │         │ • Enterprise│
             │ • Simple    │         │ • Complex   │
             │ • No server │         │ • Scaling   │
             └─────────────┘         └─────────────┘
```

## Comparison Table

| Feature | InMemory | Turso 🔥 | Neo4j |
|---------|----------|----------|-------|
| **Persistence** | ❌ | ✅ SQLite file | ✅ Server |
| **Works Offline** | ✅ | ✅ | ❌ |
| **Vector Search** | Python | Native `F32_BLOB` | Native HNSW |
| **Cloud Sync** | ❌ | ✅ Optional | ✅ |
| **Setup Complexity** | None | One file path | Server required |
| **Multi-hop Queries** | ✅ | ✅ | ✅ Native Cypher |
| **PageRank** | ✅ | ✅ | ✅ |
| **Temporal Validity** | ✅ | ✅ | ✅ |
| **Multi-tenant** | ✅ | ✅ | ✅ |
| **Best For** | Dev/Test | Edge/Offline | Enterprise |

---

## InMemory (Default)

Zero configuration, data lost on restart.

```python
from graphmem import GraphMem, MemoryConfig

config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    # No storage config = InMemory
)

memory = GraphMem(config)
```

!!! warning "Data Loss"
    InMemory storage loses all data when your program exits. Use only for development and testing.

---

## Turso (SQLite) - Recommended

SQLite-based persistence with optional cloud sync.

### Installation

```bash
pip install "agentic-graph-mem[libsql]"
```

### Local File

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # Just add a file path!
    turso_db_path="my_agent_memory.db",
)

memory = GraphMem(config)
```

### With Cloud Sync

```python
config = MemoryConfig(
    # ... LLM config ...
    
    # Local file with cloud backup
    turso_db_path="my_agent_memory.db",
    turso_url="https://your-db.turso.io",
    turso_auth_token="your-token",
)
```

!!! success "Why Turso?"
    - ✅ Data survives restarts
    - ✅ Works offline
    - ✅ No server to manage
    - ✅ Native vector search (~10ms)
    - ✅ Optional cloud sync

---

## Neo4j (Enterprise)

Full graph database for complex queries and scaling.

### Installation

```bash
pip install "agentic-graph-mem[neo4j]"
```

### Configuration

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # Neo4j connection
    neo4j_uri="neo4j+s://xxx.databases.neo4j.io",
    neo4j_username="neo4j",
    neo4j_password="your-password",
)
```

### Neo4j Aura (Cloud)

1. Create a database at [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Copy connection details
3. Use in your config

!!! tip "When to Use Neo4j"
    - Complex multi-hop graph queries
    - Need for Cypher query language
    - Horizontal scaling requirements
    - ACID transaction guarantees

---

## Redis Cache

Add high-performance caching to any storage backend.

### Installation

```bash
pip install "agentic-graph-mem[redis]"
```

### Configuration

```python
config = MemoryConfig(
    # ... LLM and storage config ...
    
    # Add Redis caching
    redis_url="redis://default:password@host:port",
)
```

### What Gets Cached

| Cache Type | TTL | Purpose |
|------------|-----|---------|
| Query results | 5 min | Instant repeat queries |
| Embeddings | 24 hours | Skip redundant API calls |
| Search results | 5 min | Fast semantic search |

### Multi-Tenant Isolation

Redis cache keys are scoped by `user_id`:

```
query:alice:chat_1:abc123  # Alice's cache
query:bob:chat_1:xyz789    # Bob's cache (isolated)
```

---

## Combined Configurations

### Development

```python
config = MemoryConfig(
    # Just LLM config, no storage
)
```

### Personal/Edge

```python
config = MemoryConfig(
    turso_db_path="memory.db",
)
```

### Production

```python
config = MemoryConfig(
    neo4j_uri="neo4j+s://...",
    neo4j_username="neo4j",
    neo4j_password="...",
    redis_url="redis://...",
)
```

---

## Migration Between Backends

GraphMem uses the same data model across all backends, making migration straightforward:

```python
# Export from one backend
old_memory = GraphMem(old_config)
nodes = old_memory.get_all_nodes()
edges = old_memory.get_all_edges()

# Import to new backend
new_memory = GraphMem(new_config)
for node in nodes:
    new_memory.store.save_node(node)
for edge in edges:
    new_memory.store.save_edge(edge)
```


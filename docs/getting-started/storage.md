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

SQLite-based persistence with optional cloud sync. Turso uses an **offline-first architecture** where all data is stored locally with optional cloud synchronization.

### Installation

```bash
pip install "agentic-graph-mem[libsql]"
```

### How Turso Storage Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    TURSO ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   LOCAL SQLITE FILE (REQUIRED)          TURSO CLOUD (OPTIONAL)  │
│   ─────────────────────────────         ────────────────────    │
│                                                                  │
│   ┌─────────────────────┐               ┌─────────────────────┐ │
│   │  my_agent.db        │    ◄─────►    │  Turso Cloud        │ │
│   │                     │     SYNC      │                     │ │
│   │  • All reads/writes │               │  • Backup           │ │
│   │  • ~1ms latency     │               │  • Multi-device     │ │
│   │  • Works offline    │               │  • Team sharing     │ │
│   └─────────────────────┘               └─────────────────────┘ │
│                                                                  │
│   turso_db_path="my_agent.db"           turso_url="libsql://..."│
│   (REQUIRED)                            turso_auth_token="..."  │
│                                         (OPTIONAL)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

!!! warning "Important: `turso_db_path` is REQUIRED"
    You **must** provide `turso_db_path` for Turso to be used. Just providing `turso_url` and `turso_auth_token` without `turso_db_path` will **NOT** enable Turso storage - it will fall back to InMemory!

### Local-Only Mode

For local persistence without cloud sync:

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ REQUIRED: Local SQLite file path
    turso_db_path="my_agent_memory.db",
)

memory = GraphMem(config)
# Data persists in my_agent_memory.db
```

### With Cloud Sync (Recommended for Production)

For local persistence **plus** automatic cloud backup and multi-device sync:

```python
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    
    # ✅ REQUIRED: Local SQLite file (always created)
    turso_db_path="my_agent_memory.db",
    
    # ✅ OPTIONAL: Cloud sync (syncs local ↔ cloud)
    turso_url="libsql://your-db-name.turso.io",
    turso_auth_token="eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9...",
)

memory = GraphMem(config)
# Data stored locally AND synced to cloud
```

### Sync Behavior

| Configuration | Behavior |
|---------------|----------|
| `turso_db_path` only | Local SQLite only, no cloud sync |
| `turso_db_path` + `turso_url` + `turso_auth_token` | Local SQLite syncs bidirectionally with Turso Cloud |
| `turso_url` + `turso_auth_token` only (NO `turso_db_path`) | ⚠️ **Turso NOT used!** Falls back to InMemory |

### Benefits of Cloud Sync

| Feature | Local Only | With Cloud Sync |
|---------|-----------|-----------------|
| Persistence | ✅ | ✅ |
| Works offline | ✅ | ✅ |
| Backup | ❌ Manual | ✅ Automatic |
| Multi-device | ❌ | ✅ |
| Team sharing | ❌ | ✅ |
| Disaster recovery | ❌ | ✅ |

### Setting Up Turso Cloud

1. **Create account**: [turso.tech](https://turso.tech)
2. **Install CLI**: `curl -sSfL https://get.tur.so/install.sh | bash`
3. **Create database**:
   ```bash
   turso db create graphmem
   turso db show graphmem --url  # Get your turso_url
   turso db tokens create graphmem  # Get your turso_auth_token
   ```

!!! success "Why Turso?"
    - ✅ Data survives restarts (local SQLite)
    - ✅ Works completely offline
    - ✅ No server to manage
    - ✅ Native vector search (~10ms)
    - ✅ Optional cloud sync for backup/multi-device
    - ✅ Offline-first: Cloud is enhancement, not requirement

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


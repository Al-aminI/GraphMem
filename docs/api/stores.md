# Storage Backends

GraphMem supports multiple storage backends.

## InMemoryStore (Default)

In-memory storage for development.

```python
from graphmem.stores.memory_store import InMemoryStore

store = InMemoryStore()
```

### Methods

- `save_node(node)` - Save a node
- `get_node(id)` - Get node by ID
- `delete_node(id)` - Delete a node
- `save_edge(edge)` - Save an edge
- `get_edges(source_id)` - Get edges from node
- `vector_search(embedding, k)` - Similarity search

---

## TursoStore

SQLite-based persistent storage with native vector search.

```python
from graphmem.stores.turso_store import TursoStore

# Local file
store = TursoStore(db_path="memory.db")

# With cloud sync
store = TursoStore(
    db_path="memory.db",
    sync_url="https://your-db.turso.io",
    auth_token="your-token",
)
```

### Methods

All methods from InMemoryStore, plus:

- `sync()` - Sync with Turso cloud
- `close()` - Close connection

### Vector Search

```python
# Native F32_BLOB vector search
results = store.vector_search(
    embedding=query_embedding,
    k=10,
    user_id="alice",
    memory_id="chat",
)
```

---

## Neo4jStore

Full graph database for enterprise.

```python
from graphmem.stores.neo4j_store import Neo4jStore

store = Neo4jStore(
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j",
    password="your-password",
)
```

### Methods

All methods from InMemoryStore, plus:

- `execute_cypher(query, params)` - Execute Cypher query
- `create_vector_index()` - Create HNSW index
- `query_edges_at_time(time)` - Temporal query
- `close()` - Close connection

### Vector Search

```python
# HNSW vector search (~5ms)
results = store.vector_search(
    embedding=query_embedding,
    k=10,
    user_id="alice",
    memory_id="chat",
)
```

### Temporal Queries

```python
# Get edges valid at a specific time
edges = store.query_edges_at_time(
    memory_id="chat",
    query_time=datetime(2015, 6, 1),
    relation_type="CEO_OF",
)
```

---

## RedisCache

Caching layer for performance.

```python
from graphmem.stores.redis_cache import RedisCache

cache = RedisCache(url="redis://localhost:6379")
```

### Methods

- `get(key)` - Get cached value
- `set(key, value, ttl)` - Set with TTL
- `delete(key)` - Delete key
- `clear_user(user_id)` - Clear user's cache
- `ping()` - Health check

### Cache Keys

```
query:{user_id}:{memory_id}:{hash}    # Query cache
search:{user_id}:{memory_id}:{hash}   # Search cache
embed:{text_hash}                      # Embedding cache (shared)
```

---

## Storage Selection

```python
from graphmem import GraphMem, MemoryConfig

# InMemory (default)
config = MemoryConfig(...)
memory = GraphMem(config)  # Uses InMemoryStore

# Turso
config = MemoryConfig(
    ...,
    turso_db_path="memory.db",
)
memory = GraphMem(config)  # Uses TursoStore

# Neo4j
config = MemoryConfig(
    ...,
    neo4j_uri="neo4j+s://...",
    neo4j_password="...",
)
memory = GraphMem(config)  # Uses Neo4jStore

# Neo4j + Redis
config = MemoryConfig(
    ...,
    neo4j_uri="neo4j+s://...",
    neo4j_password="...",
    redis_url="redis://...",
)
memory = GraphMem(config)  # Uses Neo4jStore + RedisCache
```


# Best Practices

Production tips for building robust AI agents with GraphMem.

## Ingestion Best Practices

### 1. Structure Your Data

```python
# ❌ BAD: Unstructured dumps
memory.ingest("stuff happened today blah blah...")

# ✅ GOOD: Clear, factual statements
memory.ingest("""
    On January 15, 2024, Acme Corp announced Q4 earnings:
    - Revenue: $5.2 billion (up 23% YoY)
    - CEO Jane Smith attributed growth to AI products
""")
```

### 2. Use Batch Ingestion

```python
# ❌ BAD: Sequential ingestion
for doc in documents:
    memory.ingest(doc["content"])  # Slow!

# ✅ GOOD: Batch ingestion
memory.ingest_batch(
    documents,
    max_workers=20,
    aggressive=True,
)
```

### 3. Set Appropriate Importance

```python
from graphmem import MemoryImportance

# Critical information (never decays)
memory.ingest(
    "Customer's allergies: peanuts, shellfish",
    importance=MemoryImportance.CRITICAL,
)

# Normal information (may decay over time)
memory.ingest(
    "Customer mentioned they like coffee",
    importance=MemoryImportance.LOW,
)
```

### 4. Include Temporal Context

```python
# ❌ BAD: No temporal context
memory.ingest("The CEO is John Smith")

# ✅ GOOD: Include dates
memory.ingest("As of January 2024, the CEO is John Smith")
```

---

## Evolution Best Practices

### 1. Evolve Regularly

```python
# Option 1: Auto-evolve
config = MemoryConfig(..., auto_evolve=True)

# Option 2: After batch ingestion
memory.ingest_batch(docs)
memory.evolve()

# Option 3: On a schedule
import schedule
schedule.every().day.at("02:00").do(memory.evolve)
```

### 2. Configure Decay Appropriately

```python
config = MemoryConfig(
    decay_enabled=True,
    decay_half_life_days=30,  # Adjust based on domain
    # Shorter for chat (7 days)
    # Longer for knowledge bases (90 days)
)
```

### 3. Tune Consolidation Threshold

```python
config = MemoryConfig(
    consolidation_threshold=0.85,
    # Higher (0.9+): Only merge very similar entities
    # Lower (0.7-): More aggressive merging
)
```

---

## Query Best Practices

### 1. Handle Low Confidence

```python
response = memory.query(question)

if response.confidence < 0.5:
    # Escalate or provide fallback
    return "I'm not sure. Let me check with a human."
elif response.confidence < 0.7:
    return f"I think: {response.answer} (but I'm not certain)"
else:
    return response.answer
```

### 2. Use Response Context

```python
response = memory.query(question)

# Use context for follow-up prompts
context = response.context
sources = [n.name for n in response.nodes]

# Log for debugging
print(f"Answer based on: {sources}")
```

---

## Storage Best Practices

### 1. Choose the Right Backend

| Use Case | Recommended |
|----------|-------------|
| Development | InMemory |
| Personal/Edge | Turso |
| Production | Neo4j + Redis |
| Enterprise | Neo4j Aura + Redis Cloud |

### 2. Use Redis Caching

```python
config = MemoryConfig(
    redis_url="redis://...",  # Add caching for production
)
```

### 3. Per-User Database Files (Turso)

```python
# Maximum isolation
config = MemoryConfig(
    turso_db_path=f"memories/{user_id}.db",
)
```

---

## Error Handling

### 1. Handle Ingestion Errors

```python
from graphmem.core.exceptions import IngestionError

try:
    memory.ingest(content)
except IngestionError as e:
    logger.error(f"Ingestion failed: {e}")
    # Queue for retry or manual review
```

### 2. Handle Query Errors

```python
from graphmem.core.exceptions import QueryError

try:
    response = memory.query(question)
except QueryError as e:
    logger.error(f"Query failed: {e}")
    return "I'm having trouble answering. Please try again."
```

### 3. Retry with Backoff

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def safe_ingest(memory, content):
    return memory.ingest(content)
```

---

## Performance Optimization

### 1. Batch Operations

```python
# Batch ingestion with optimal workers
memory.ingest_batch(
    documents,
    max_workers=20,
    aggressive=True,
    show_progress=True,
)
```

### 2. Defer Community Building

```python
# For large batch ingestion
for batch in batches:
    memory.ingest_batch(batch, rebuild_communities=False)

# Build communities once at the end
memory.evolve()
```

### 3. Cache Frequently Accessed Data

```python
# Redis caching automatically handles this
config = MemoryConfig(
    redis_url="redis://...",
)
# Repeated queries are instant
```

---

## Security Best Practices

### 1. Never Trust Client User ID

```python
# ❌ BAD: Client-provided user_id
@app.post("/query")
async def query(user_id: str, question: str):  # Untrusted!
    memory = get_memory(user_id)
    ...

# ✅ GOOD: Extract from verified token
@app.post("/query")
async def query(
    question: str,
    user_id: str = Depends(get_current_user_from_token),
):
    memory = get_memory(user_id)
    ...
```

### 2. Use Environment Variables

```python
import os

config = MemoryConfig(
    llm_api_key=os.getenv("OPENAI_API_KEY"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    redis_url=os.getenv("REDIS_URL"),
)
```

### 3. Audit Access

```python
import logging

logger = logging.getLogger("graphmem.audit")

def query_with_audit(user_id: str, question: str):
    logger.info(f"Query: user={user_id}, question={question[:50]}...")
    response = memory.query(question)
    logger.info(f"Response: user={user_id}, confidence={response.confidence}")
    return response
```

---

## Monitoring

### 1. Track Key Metrics

```python
# After ingestion
result = memory.ingest(content)
metrics.gauge("graphmem.entities", result["entities"])
metrics.gauge("graphmem.relationships", result["relationships"])

# After query
response = memory.query(question)
metrics.histogram("graphmem.query_latency", response.latency_ms)
metrics.gauge("graphmem.query_confidence", response.confidence)

# After evolution
events = memory.evolve()
metrics.counter("graphmem.evolution_events", len(events))
```

### 2. Set Up Alerts

```yaml
# Example Prometheus alerts
groups:
  - name: graphmem
    rules:
      - alert: LowQueryConfidence
        expr: avg(graphmem_query_confidence) < 0.5
        for: 5m
        
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, graphmem_query_latency) > 5000
        for: 5m
```


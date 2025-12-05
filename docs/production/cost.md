# Cost Optimization

Optimize GraphMem costs for production.

## Cost Breakdown

### Typical Monthly Costs

| Component | Cloud Service | Self-Hosted |
|-----------|--------------|-------------|
| **Neo4j** | $500-2,500 (Aura) | $300-500 (EC2) |
| **Redis** | $200-500 (ElastiCache) | $100-200 (EC2) |
| **LLM API** | $500-5,000 | $500-5,000 |
| **Embeddings API** | $100-1,000 | $0-100 (local) |
| **Compute** | $300-1,000 | $200-500 |
| **Total** | $1,600-10,000 | $1,100-6,300 |

---

## Optimization Strategies

### 1. Use Turso Instead of Neo4j

For smaller deployments, Turso offers:
- Free tier available
- No server costs
- Native vector search

```python
config = MemoryConfig(
    turso_db_path="memory.db",  # Local file, zero cost
    # OR
    turso_url="https://your-db.turso.io",  # Cloud, low cost
)
```

**Savings**: $200-2,500/month

### 2. Use Local Embeddings

Replace API-based embeddings with local models:

```python
# Install sentence-transformers
pip install sentence-transformers

# Use local model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 100x cheaper, faster for high volume
```

**Savings**: $100-1,000/month (at scale)

### 3. Use GPT-4o-mini Instead of GPT-4

```python
config = MemoryConfig(
    llm_model="gpt-4o-mini",  # 10x cheaper than gpt-4
)
```

**Savings**: 90% on LLM costs

### 4. Redis Caching

Reduces redundant API calls:

```python
config = MemoryConfig(
    redis_url="redis://...",  # Cache queries and embeddings
)
```

**Impact**: 50-80% reduction in API calls

### 5. Batch Ingestion

Reduces per-document overhead:

```python
# ❌ Expensive: Sequential ingestion
for doc in documents:
    memory.ingest(doc)

# ✅ Cheaper: Batch ingestion
memory.ingest_batch(documents, max_workers=20)
```

**Impact**: 3-5x faster, lower API costs

---

## Cost Calculator

### Per 1M Documents

| Step | OpenAI | Azure | Local |
|------|--------|-------|-------|
| Embeddings | $130 | $130 | $10 |
| Extraction | $2,000 | $2,000 | $2,000 |
| Storage | $50 | $50 | $20 |
| **Total** | $2,180 | $2,180 | $2,030 |

### Per 1M Queries

| Component | Cached | Uncached |
|-----------|--------|----------|
| Retrieval | $0 | $50 |
| LLM Answer | $200 | $200 |
| **Total** | $200 | $250 |

---

## Budget Tiers

### Startup (< $500/month)

```python
config = MemoryConfig(
    llm_model="gpt-4o-mini",
    turso_db_path="memory.db",
    # No Redis (use in-memory)
)
```

### Growth ($500-2,000/month)

```python
config = MemoryConfig(
    llm_model="gpt-4o-mini",
    neo4j_uri="neo4j+s://...",  # Aura Free/Basic
    redis_url="redis://...",    # Small Redis
)
```

### Enterprise ($2,000+/month)

```python
config = MemoryConfig(
    llm_model="gpt-4o",         # Best quality
    neo4j_uri="neo4j+s://...",  # Aura Professional
    redis_url="redis://...",    # Redis Enterprise
)
```

---

## Monitoring Costs

### Track API Usage

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Track per query
response = memory.query(question)
tokens_used = count_tokens(response.context)
cost = tokens_used * 0.00015 / 1000  # GPT-4o-mini pricing

# Log for monitoring
logger.info(f"Query cost: ${cost:.6f}")
```

### Set Spending Limits

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    # Set spending limit in OpenAI dashboard
)
```

---

## Free Tier Options

| Service | Free Tier |
|---------|-----------|
| **Turso** | 9GB storage, 500M rows read |
| **Neo4j Aura** | Free tier available |
| **Redis Cloud** | 30MB free |
| **OpenAI** | $5 credit for new users |
| **Azure OpenAI** | Pay-as-you-go |

---

## Summary

1. **Start with Turso** - Free/cheap, works offline
2. **Use GPT-4o-mini** - 10x cheaper than GPT-4
3. **Enable Redis caching** - 50-80% fewer API calls
4. **Batch operations** - Lower overhead
5. **Monitor usage** - Track tokens and costs


# Production Architecture

Deploy GraphMem at scale with confidence.

## Recommended Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    APPLICATION LAYER                      │    │
│  │                                                          │    │
│  │   FastAPI / Flask / Django                               │    │
│  │       ↓                                                  │    │
│  │   GraphMem Instance (per request or singleton)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     STORAGE LAYER                         │    │
│  │                                                          │    │
│  │  Neo4j Aura (Graph)  ←→  Redis Cloud (Cache)            │    │
│  │         ↑                        ↑                       │    │
│  │         └────────────────────────┘                       │    │
│  │                    ↓                                     │    │
│  │            Turso (Backup/Vectors)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LLM PROVIDERS                          │    │
│  │                                                          │    │
│  │  OpenAI  |  Azure OpenAI  |  Anthropic  |  Local LLMs   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Selection

### By Scale

| Scale | Users | Recommendation |
|-------|-------|----------------|
| **Development** | 1 | InMemory |
| **Personal/Edge** | 1-10 | Turso |
| **Startup** | 1-100 | Turso + Cloud sync |
| **Growth** | 100-10K | Neo4j + Redis |
| **Enterprise** | 10K+ | Neo4j Enterprise + Redis Cluster |

### Component Matrix

| Component | Development | Production |
|-----------|-------------|------------|
| **Storage** | InMemory | Neo4j Aura |
| **Cache** | None | Redis Cloud |
| **LLM** | OpenAI | Azure OpenAI |
| **Embeddings** | OpenAI | Azure OpenAI |
| **Compute** | Local | Kubernetes |

---

## FastAPI Example

```python
from fastapi import FastAPI, Depends
from functools import lru_cache
import os

app = FastAPI()

@lru_cache()
def get_memory() -> GraphMem:
    config = MemoryConfig(
        llm_provider="azure",
        llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        azure_deployment="gpt-4",
        llm_model="gpt-4",
        
        embedding_provider="azure",
        embedding_api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_embedding_deployment="text-embedding-ada-002",
        embedding_model="text-embedding-ada-002",
        
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_username="neo4j",
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        
        redis_url=os.getenv("REDIS_URL"),
        
        evolution_enabled=True,
    )
    return GraphMem(config)

@app.post("/ingest")
async def ingest(content: str, memory: GraphMem = Depends(get_memory)):
    result = memory.ingest(content)
    return {"entities": result["entities"]}

@app.post("/query")
async def query(question: str, memory: GraphMem = Depends(get_memory)):
    response = memory.query(question)
    return {"answer": response.answer, "confidence": response.confidence}

@app.post("/evolve")
async def evolve(memory: GraphMem = Depends(get_memory)):
    events = memory.evolve()
    return {"events": len(events)}
```

---

## Environment Configuration

```bash
# .env.production
# LLM
AZURE_OPENAI_KEY=your-key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/

# Storage
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_PASSWORD=your-password

# Cache
REDIS_URL=redis://default:password@host:port

# Evolution
EVOLUTION_ENABLED=true
AUTO_EVOLVE=false
DECAY_HALF_LIFE_DAYS=30
```

```python
import os
from dotenv import load_dotenv

load_dotenv(".env.production")

config = MemoryConfig(
    llm_provider="azure",
    llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
    # ... rest of config
)
```

---

## High Availability

### Neo4j Cluster

```
┌─────────────────────────────────────────┐
│           NEO4J CLUSTER                  │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ Primary  │  │ Replica  │  │ Replica  │
│  │  (Write) │  │  (Read)  │  │  (Read)  │
│  └──────────┘  └──────────┘  └──────────┘
│                                          │
│  Load Balancer routes reads to replicas  │
└─────────────────────────────────────────┘
```

### Redis Cluster

```
┌─────────────────────────────────────────┐
│           REDIS CLUSTER                  │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │ Master 1 │  │ Master 2 │  │ Master 3 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘
│       │             │             │      │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
│  │ Replica  │  │ Replica  │  │ Replica  │
│  └──────────┘  └──────────┘  └──────────┘
└─────────────────────────────────────────┘
```

---

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Neo4j connection tested
- [ ] Redis connection tested
- [ ] LLM API verified
- [ ] Health checks implemented
- [ ] Logging configured
- [ ] Metrics collection enabled
- [ ] Alerts set up
- [ ] Backup strategy defined
- [ ] Rate limiting configured


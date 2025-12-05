# Distributed Infrastructure

Scale GraphMem to handle millions of documents and thousands of concurrent users.

!!! tip "Full Guide Available"
    For a complete distributed infrastructure guide with code examples, see the [DISTRIBUTED_INFRASTRUCTURE.md](https://github.com/Al-aminI/GraphMem/blob/main/docs/DISTRIBUTED_INFRASTRUCTURE.md) in the repository.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DISTRIBUTED GRAPHMEM                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐   │
│  │   PRODUCERS    │────▶│   MESSAGE QUEUE     │────▶│    WORKER POOL       │   │
│  │                │     │   (Redpanda/Kafka)  │     │                      │   │
│  │ • Datasets     │     │                     │     │  ┌──────┐ ┌──────┐  │   │
│  │ • APIs         │     │  • ingest_queue     │     │  │Worker│ │Worker│  │   │
│  │ • Files        │     │  • embed_queue      │     │  │  1   │ │  2   │  │   │
│  └────────────────┘     │  • extract_queue    │     │  └──────┘ └──────┘  │   │
│                         └─────────────────────┘     └──────────────────────┘   │
│                                    │                          │                 │
│                                    ▼                          ▼                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                          PROCESSING LAYER                                 │  │
│  │                                                                           │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │  │
│  │  │ EMBEDDING POOL  │  │ LLM EXTRACTION  │  │   GRAPH OPERATIONS      │  │  │
│  │  │                 │  │                 │  │                         │  │  │
│  │  │ • GPU Workers   │  │ • Async Batch   │  │ • Entity Resolution     │  │  │
│  │  │ • Batch 1000+   │  │ • Rate Limiting │  │ • Community Detection   │  │  │
│  │  │ • Local Models  │  │ • Retry Logic   │  │ • Evolution             │  │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           STORAGE LAYER                                   │  │
│  │                                                                           │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │  │
│  │  │   Neo4j     │    │    Redis    │    │   Turso     │                  │  │
│  │  │  Cluster    │    │   Cluster   │    │  (Backup)   │                  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

| Metric | Target | How |
|--------|--------|-----|
| **Ingestion** | 10,000 docs/sec | GPU embeddings + async LLM |
| **Query p50** | < 100ms | Redis cache + read replicas |
| **Query p99** | < 500ms | Connection pooling |
| **Concurrent Users** | 10,000+ | Horizontal scaling |
| **Document Capacity** | 100M+ | Neo4j sharding |

---

## Key Components

### Message Queue (Redpanda/Kafka)

Decouples producers from consumers for reliability and scaling:

```yaml
topics:
  - graphmem.ingest.raw      # Incoming documents
  - graphmem.ingest.embed    # For embedding
  - graphmem.ingest.extract  # For LLM extraction
  - graphmem.graph.resolve   # Entity resolution
  - graphmem.query.pending   # Query queue
```

### GPU Embedding Cluster

Local GPU models for high-throughput embedding:

- Single A100: ~5,000 embeddings/sec
- Single H100: ~10,000 embeddings/sec
- 100x faster than API calls

### Worker Pool

Horizontally scalable workers:

- **Embed Workers**: GPU-accelerated embedding
- **Extract Workers**: LLM-based extraction
- **Graph Workers**: Entity resolution, evolution
- **Query Workers**: Query processing

---

## Docker Compose (Development)

```yaml
version: '3.8'

services:
  redpanda:
    image: vectorized/redpanda:latest
    ports:
      - "9092:9092"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  neo4j:
    image: neo4j:5-enterprise
    ports:
      - "7474:7474"
      - "7687:7687"

  gateway:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redpanda
      - redis
      - neo4j

  embed-worker:
    build: .
    command: ["--type", "embed"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  extract-worker:
    build: .
    command: ["--type", "extract"]
    deploy:
      replicas: 5
```

---

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphmem-worker
spec:
  replicas: 10
  template:
    spec:
      containers:
      - name: worker
        image: graphmem/worker:latest
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graphmem-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: graphmem-worker
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: kafka_consumer_lag
      target:
        type: AverageValue
        averageValue: "1000"
```

---

## See Also

- [Full Distributed Infrastructure Guide](https://github.com/Al-aminI/GraphMem/blob/main/docs/DISTRIBUTED_INFRASTRUCTURE.md)
- [Deployment Guide](deployment.md)
- [Monitoring](monitoring.md)


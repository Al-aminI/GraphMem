# Monitoring & Observability

Monitor GraphMem in production.

## Key Metrics

### Ingestion Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `graphmem_docs_ingested_total` | Total documents ingested | - |
| `graphmem_ingestion_latency_seconds` | Ingestion latency | p99 > 30s |
| `graphmem_entities_extracted_total` | Entities extracted | - |
| `graphmem_relationships_extracted_total` | Relationships extracted | - |

### Query Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `graphmem_queries_total` | Total queries processed | - |
| `graphmem_query_latency_seconds` | Query latency | p99 > 5s |
| `graphmem_query_confidence` | Answer confidence | avg < 0.5 |

### Cache Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `graphmem_cache_hits_total` | Cache hits | - |
| `graphmem_cache_misses_total` | Cache misses | hit rate < 50% |

---

## Prometheus Integration

### FastAPI Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI, Response

app = FastAPI()

# Define metrics
QUERIES_TOTAL = Counter(
    'graphmem_queries_total',
    'Total queries',
    ['tenant_id', 'status']
)

QUERY_LATENCY = Histogram(
    'graphmem_query_latency_seconds',
    'Query latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

QUERY_CONFIDENCE = Histogram(
    'graphmem_query_confidence',
    'Query confidence',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

@app.post("/query")
async def query(question: str, tenant_id: str):
    with QUERY_LATENCY.time():
        response = memory.query(question)
    
    QUERIES_TOTAL.labels(tenant_id=tenant_id, status="success").inc()
    QUERY_CONFIDENCE.observe(response.confidence)
    
    return {"answer": response.answer}

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Prometheus Config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'graphmem-api'
    static_configs:
      - targets: ['graphmem-api:8000']
    metrics_path: /metrics
```

---

## Grafana Dashboard

### Query Performance Panel

```json
{
  "title": "Query Latency (p99)",
  "type": "graph",
  "targets": [
    {
      "expr": "histogram_quantile(0.99, rate(graphmem_query_latency_seconds_bucket[5m]))",
      "legendFormat": "p99"
    },
    {
      "expr": "histogram_quantile(0.50, rate(graphmem_query_latency_seconds_bucket[5m]))",
      "legendFormat": "p50"
    }
  ]
}
```

### Cache Hit Rate Panel

```json
{
  "title": "Cache Hit Rate",
  "type": "gauge",
  "targets": [
    {
      "expr": "sum(rate(graphmem_cache_hits_total[5m])) / (sum(rate(graphmem_cache_hits_total[5m])) + sum(rate(graphmem_cache_misses_total[5m])))"
    }
  ]
}
```

---

## Alerting

### Prometheus Alerts

```yaml
groups:
  - name: graphmem
    rules:
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, rate(graphmem_query_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High query latency"
          
      - alert: LowQueryConfidence
        expr: avg(graphmem_query_confidence) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low answer confidence"
          
      - alert: LowCacheHitRate
        expr: |
          sum(rate(graphmem_cache_hits_total[5m])) / 
          (sum(rate(graphmem_cache_hits_total[5m])) + sum(rate(graphmem_cache_misses_total[5m]))) < 0.5
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Cache hit rate below 50%"
```

---

## Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, 'tenant_id'):
            log_obj['tenant_id'] = record.tenant_id
        if hasattr(record, 'query'):
            log_obj['query'] = record.query
        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

### Log Aggregation (ELK/Loki)

```yaml
# Loki configuration
scrape_configs:
  - job_name: graphmem
    static_configs:
      - targets:
          - localhost
        labels:
          job: graphmem
          __path__: /var/log/graphmem/*.log
```

---

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize tracing
tracer = trace.get_tracer(__name__)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

@app.post("/query")
async def query(question: str):
    with tracer.start_as_current_span("graphmem.query") as span:
        span.set_attribute("question_length", len(question))
        
        response = memory.query(question)
        
        span.set_attribute("confidence", response.confidence)
        span.set_attribute("entities_found", len(response.nodes))
        
        return {"answer": response.answer}
```

---

## Health Dashboards

### Key Panels

1. **Ingestion Rate** - Documents/minute
2. **Query Latency** - p50, p95, p99
3. **Cache Hit Rate** - Percentage
4. **Error Rate** - Errors/minute
5. **Active Tenants** - Unique users
6. **Entity Count** - Graph size
7. **Evolution Events** - Per day

### Example Dashboard JSON

```json
{
  "dashboard": {
    "title": "GraphMem Production",
    "panels": [
      {
        "title": "Ingestion Rate",
        "type": "stat",
        "targets": [{"expr": "rate(graphmem_docs_ingested_total[5m])"}]
      },
      {
        "title": "Query Latency",
        "type": "graph",
        "targets": [
          {"expr": "histogram_quantile(0.99, rate(graphmem_query_latency_seconds_bucket[5m]))"}
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [{"expr": "cache_hit_rate"}]
      }
    ]
  }
}
```


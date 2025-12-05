# Deployment Guide

Deploy GraphMem to production environments.

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```
agentic-graph-mem[all]
fastapi
uvicorn
python-dotenv
```

### Build and Run

```bash
# Build
docker build -t graphmem-api .

# Run
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e NEO4J_URI=$NEO4J_URI \
  -e NEO4J_PASSWORD=$NEO4J_PASSWORD \
  -e REDIS_URL=$REDIS_URL \
  graphmem-api
```

---

## Kubernetes Deployment

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphmem-api
  labels:
    app: graphmem-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: graphmem-api
  template:
    metadata:
      labels:
        app: graphmem-api
    spec:
      containers:
      - name: api
        image: graphmem-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: graphmem-secrets
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: graphmem-api
spec:
  selector:
    app: graphmem-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: graphmem-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  NEO4J_URI: "neo4j+s://..."
  NEO4J_PASSWORD: "..."
  REDIS_URL: "redis://..."
```

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: graphmem-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: graphmem-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Cloud Platform Guides

### AWS

```bash
# ECR
aws ecr create-repository --repository-name graphmem-api
docker tag graphmem-api:latest $ECR_URI/graphmem-api:latest
docker push $ECR_URI/graphmem-api:latest

# EKS
eksctl create cluster --name graphmem --region us-east-1
kubectl apply -f k8s/
```

### Google Cloud

```bash
# GCR
docker tag graphmem-api gcr.io/$PROJECT_ID/graphmem-api
docker push gcr.io/$PROJECT_ID/graphmem-api

# GKE
gcloud container clusters create graphmem --zone us-central1-a
kubectl apply -f k8s/
```

### Azure

```bash
# ACR
az acr create --name graphmemacr --resource-group rg-graphmem --sku Basic
az acr login --name graphmemacr
docker tag graphmem-api graphmemacr.azurecr.io/graphmem-api
docker push graphmemacr.azurecr.io/graphmem-api

# AKS
az aks create --name graphmem --resource-group rg-graphmem
kubectl apply -f k8s/
```

---

## Health Checks

### Implementation

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    """Liveness probe - is the service running?"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness probe - can we serve traffic?"""
    try:
        # Check Neo4j connection
        memory.store.health_check()
        # Check Redis connection
        memory.cache.ping()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "error": str(e)}, 503
```

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...           # Or Azure credentials
NEO4J_URI=neo4j+s://...         # Neo4j connection
NEO4J_PASSWORD=...              # Neo4j password

# Optional
REDIS_URL=redis://...           # Redis for caching
LOG_LEVEL=INFO                  # Logging level
EVOLUTION_ENABLED=true          # Enable evolution
AUTO_EVOLVE=false               # Auto-evolve on ingest
DECAY_HALF_LIFE_DAYS=30         # Decay configuration
```

---

## SSL/TLS Configuration

### With Let's Encrypt (Kubernetes)

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: graphmem-tls
spec:
  secretName: graphmem-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.yourdomain.com
```

### Ingress with TLS

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: graphmem-ingress
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: graphmem-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: graphmem-api
            port:
              number: 80
```


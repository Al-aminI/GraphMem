# 🧠 GraphMem

**Self-Evolving Graph-Based Memory for Production AI Agents**

[![PyPI](https://img.shields.io/pypi/v/agentic-graph-mem.svg)](https://pypi.org/project/agentic-graph-mem/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-Al--aminI/GraphMem-blue.svg)](https://github.com/Al-aminI/GraphMem)

GraphMem is a state-of-the-art, self-evolving graph-based memory system for production AI agents. It achieves **99% token reduction**, **4.2× faster queries**, and **bounded memory growth** compared to naive RAG approaches.

## 📊 Benchmark Results

| Metric | Naive RAG | GraphMem | Improvement |
|--------|-----------|----------|-------------|
| **Tokens/Query** | 703 | 7 | **99% reduction** |
| **Query Latency** | 1656ms | 394ms | **4.2× faster** |
| **Entity Resolution** | 20% | 95% | **+75%** |
| **Multi-hop Reasoning** | 50-67% | 85-86% | **+35%** |
| **Long Context (100 facts)** | 0% | 90% | **+90%** |
| **Memory Growth (1 year)** | 3,650 | ~100 | **97% bounded** |

## ✨ Key Features

### 🔄 Self-Evolving Memory
- **Importance Scoring**: Multi-factor scoring (recency, frequency, centrality, feedback)
- **Memory Decay**: Exponential decay inspired by Ebbinghaus forgetting curve
- **Consolidation**: LLM-based merging of redundant memories (80% reduction)
- **Temporal Tracking**: Track how facts change over time

### 🕸️ Graph-Based Knowledge
- **Entity Resolution**: Hybrid lexical + semantic matching (95% accuracy)
- **Community Detection**: Automatic topic clustering with summaries
- **Multi-hop Reasoning**: Graph traversal for complex queries
- **O(1) Entity Lookup**: Direct graph indexing vs O(n) vector search

### 📚 Context Engineering
- **Semantic Chunking**: 0.90 coherence (vs 0.56 for fixed-size)
- **Relevance-Weighted Assembly**: 53% better context relevance
- **Token Optimization**: 99% reduction through targeted retrieval
- **Multi-source Synthesis**: Cross-document fact extraction

### 🚀 Production Ready
- **Neo4j Backend**: Enterprise graph database with ACID transactions
- **Redis Caching**: Sub-millisecond retrieval
- **Multi-LLM Support**: OpenAI, Azure, Anthropic, OpenRouter, Groq, Together, Ollama
- **Any OpenAI-Compatible API**: Works with 100+ models via OpenRouter, etc.
- **Scalable**: Handles 100K+ entities efficiently

## 🏁 Quick Start

### Installation

```bash
pip install agentic-graph-mem
```

### Basic Usage

```python
from graphmem import GraphMem, MemoryConfig

# Initialize with configuration
config = MemoryConfig(
    llm_provider="azure_openai",
    llm_api_key="your-api-key",
    llm_model="gpt-4",
)
memory = GraphMem(config)

# Access the memory object
print(f"Memory ID: {memory.memory_id}")
print(f"Nodes: {memory.memory.node_count}")
print(f"Edges: {memory.memory.edge_count}")
```

### Working with Memory Directly

```python
from graphmem import Memory, MemoryNode, MemoryEdge, MemoryCluster

# Create a memory object
mem = Memory(id="my_agent_memory", name="Agent Knowledge Base")

# Add entities (nodes)
mem.add_node(MemoryNode(
    id="entity_1",
    name="OpenAI",
    entity_type="Organization",
    description="AI research company that created ChatGPT",
))

mem.add_node(MemoryNode(
    id="entity_2", 
    name="Sam Altman",
    entity_type="Person",
    description="CEO of OpenAI",
))

# Add relationships (edges)
mem.add_edge(MemoryEdge(
    id="rel_1",
    source_id="entity_2",
    target_id="entity_1",
    relation_type="CEO_OF",
))

# Add community summaries
mem.add_cluster(MemoryCluster(
    id=1,
    summary="OpenAI is an AI company led by Sam Altman...",
    entities=["OpenAI", "Sam Altman"],
))

print(f"Memory has {mem.node_count} nodes, {mem.edge_count} edges")
```

### Using Storage Backends

```python
from graphmem import Neo4jStore, RedisCache, Memory

# Neo4j for persistent graph storage
neo4j = Neo4jStore(
    uri="neo4j+ssc://your-instance.databases.neo4j.io",
    username="neo4j",
    password="your-password",
)

# Save memory to Neo4j
memory = Memory(id="production_memory", name="Production KB")
# ... add nodes and edges ...
neo4j.save_memory(memory)

# Load memory from Neo4j
loaded = neo4j.load_memory("production_memory")
print(f"Loaded {loaded.node_count} nodes")

# Redis for high-speed caching
redis = RedisCache(
    url="redis://default:password@host:port",
    prefix="graphmem",
)

# Cache memory state
redis.cache_memory_state("production_memory", {
    "nodes": memory.node_count,
    "edges": memory.edge_count,
    "last_updated": "2024-01-01",
})

# Retrieve cached state
state = redis.get_memory_state("production_memory")

# Cleanup
neo4j.close()
redis.close()
```

### Using Different LLM Providers

GraphMem supports **any OpenAI-compatible API**, giving you access to 100+ models:

```python
from graphmem.llm.providers import LLMProvider, openrouter, groq, together

# OpenAI
llm = LLMProvider(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o",
)

# Azure OpenAI
llm = LLMProvider(
    provider="azure_openai",
    api_key="your-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment="gpt-4",
)

# OpenRouter (100+ models including Gemini, Claude, Llama, etc.)
llm = LLMProvider(
    provider="openai_compatible",
    api_key="sk-or-v1-...",
    api_base="https://openrouter.ai/api/v1",
    model="google/gemini-2.0-flash-001",  # or any model on OpenRouter
)

# Convenience function for OpenRouter
llm = openrouter(
    api_key="sk-or-v1-...",
    model="anthropic/claude-3.5-sonnet",
)

# Groq (ultra-fast inference)
llm = LLMProvider(
    provider="openai_compatible",
    api_key="gsk_...",
    api_base="https://api.groq.com/openai/v1",
    model="llama-3.1-70b-versatile",
)

# Together AI
llm = LLMProvider(
    provider="openai_compatible",
    api_key="...",
    api_base="https://api.together.xyz/v1",
    model="meta-llama/Llama-3-70b-chat-hf",
)

# Anthropic Claude (native)
llm = LLMProvider(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",
)

# Local Ollama
llm = LLMProvider(
    provider="ollama",
    model="llama3.2",
)

# Use it!
response = llm.complete("What is the capital of France?")
print(response)
```

### LLM-Based Knowledge Extraction

```python
from graphmem.llm.providers import LLMProvider

# Initialize LLM provider (any provider works!)
llm = LLMProvider(
    provider="openai_compatible",
    api_key="sk-or-v1-...",
    api_base="https://openrouter.ai/api/v1",
    model="google/gemini-2.0-flash-001",
)

# Extract knowledge from text
content = """
Tesla, Inc. is an electric vehicle company headquartered in Austin, Texas.
Elon Musk is the CEO of Tesla. The company produces Model S, Model 3, Model X, and Model Y.
"""

extraction_prompt = f"""Extract all entities and relationships from this text.

For each entity: ENTITY|name|type|description
For each relationship: RELATION|source|relationship|target

Text: {content}

Output:"""

result = llm.complete(extraction_prompt)
print(result)
# ENTITY|Tesla|Organization|Electric vehicle company
# ENTITY|Elon Musk|Person|CEO of Tesla
# ENTITY|Austin, Texas|Location|Headquarters of Tesla
# RELATION|Elon Musk|CEO_OF|Tesla
# RELATION|Tesla|HEADQUARTERED_IN|Austin, Texas
```

### Context Engineering

```python
from graphmem.context.chunker import DocumentChunker
from graphmem.context.context_engine import ContextEngine

# Semantic document chunking
chunker = DocumentChunker(
    chunk_size=500,
    chunk_overlap=50,
    strategy="semantic",  # or "fixed", "paragraph"
)

document = """
# Introduction to Distributed Systems

Distributed systems are collections of independent computers...
[long document]
"""

chunks = chunker.chunk(document)
print(f"Created {len(chunks)} semantic chunks")

# Context window assembly
engine = ContextEngine(max_tokens=4000)
context = engine.build_context(
    query="How does consensus work?",
    sources=chunks,
    strategy="relevance_weighted",
)
print(f"Assembled {len(context.split())} tokens of relevant context")
```

## 🏗️ Architecture

```
graphmem/
├── core/
│   ├── memory.py          # GraphMem main class
│   ├── memory_types.py    # Memory, MemoryNode, MemoryEdge, MemoryCluster
│   └── exceptions.py      # Custom exceptions
│
├── graph/
│   ├── knowledge_graph.py # Knowledge extraction & graph ops
│   ├── entity_resolver.py # Entity deduplication (95% accuracy)
│   └── community_detector.py # Topic clustering
│
├── evolution/
│   ├── memory_evolution.py # Evolution orchestrator
│   ├── importance_scorer.py # Multi-factor importance
│   ├── decay.py           # Exponential decay
│   ├── consolidation.py   # LLM-based merging
│   └── rehydration.py     # Memory restoration
│
├── retrieval/
│   ├── query_engine.py    # Query processing
│   ├── retriever.py       # Context retrieval
│   └── semantic_search.py # Embedding search
│
├── context/
│   ├── context_engine.py  # Context assembly
│   ├── chunker.py         # Semantic chunking
│   └── multimodal.py      # PDF, image, audio
│
├── llm/
│   ├── providers.py       # LLMProvider (Azure, OpenAI, Anthropic)
│   └── embeddings.py      # EmbeddingProvider
│
├── stores/
│   ├── neo4j_store.py     # Graph persistence
│   └── redis_cache.py     # High-speed caching
│
└── evaluation/
    ├── benchmarks.py      # Core benchmarks
    ├── context_engineering.py # Context eval
    └── run_evaluation.py  # Full evaluation suite
```

## 📖 Self-Evolution Mechanisms

### Importance Scoring

```python
# Importance is computed from multiple factors:
importance = (
    w1 * recency +      # exp(-λ * time_since_access)
    w2 * frequency +    # log(1 + access_count) / log(1 + max_count)
    w3 * centrality +   # PageRank score
    w4 * feedback       # explicit user signals
)

# Default weights: (0.3, 0.3, 0.2, 0.2)
```

### Memory Decay

```python
# Exponential decay inspired by Ebbinghaus forgetting curve
importance(t) = importance_0 * exp(-λ * (t - last_access))

# Entities below threshold are archived
if importance < 0.1:
    archive(entity)
```

### Consolidation

```python
# Similar memories are merged using LLM
# Before: 5 separate mentions of "user likes Python"
# After: 1 consolidated entity with merged properties

# Achieves 80% memory reduction on redundant content
```

## 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm_provider` | LLM provider (see below) | `azure_openai` |
| `llm_api_key` | API key for LLM | Required |
| `llm_api_base` | API base URL (for openai_compatible) | Provider default |
| `llm_model` | Model name/deployment | `gpt-4` |
| `embedding_provider` | Embedding provider | `azure_openai` |
| `neo4j_uri` | Neo4j connection URI | `bolt://localhost:7687` |
| `neo4j_password` | Neo4j password | Required for cloud |
| `redis_url` | Redis connection URL | `redis://localhost:6379` |
| `decay_rate` | Importance decay rate | `0.01` |
| `consolidation_threshold` | Similarity for merging | `0.85` |
| `entity_resolution_threshold` | Similarity for entity matching | `0.85` |

### Supported LLM Providers

| Provider | `llm_provider` | `api_base` |
|----------|----------------|------------|
| OpenAI | `openai` | (default) |
| Azure OpenAI | `azure_openai` | Your Azure endpoint |
| OpenRouter | `openai_compatible` | `https://openrouter.ai/api/v1` |
| Groq | `openai_compatible` | `https://api.groq.com/openai/v1` |
| Together AI | `openai_compatible` | `https://api.together.xyz/v1` |
| Fireworks | `openai_compatible` | `https://api.fireworks.ai/inference/v1` |
| Mistral | `openai_compatible` | `https://api.mistral.ai/v1` |
| DeepInfra | `openai_compatible` | `https://api.deepinfra.com/v1/openai` |
| Anthropic | `anthropic` | (default) |
| Ollama | `ollama` | `http://localhost:11434` |

## 🧪 Running Evaluations

```bash
# Install the package
pip install agentic-graph-mem

# Run benchmarks
cd graphmem/evaluation

# Set credentials
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=your-endpoint

# Run full evaluation
python run_evaluation.py --azure-endpoint $AZURE_OPENAI_ENDPOINT --azure-key $AZURE_OPENAI_API_KEY
```

## 📄 Research Paper

For full details, see our research paper:

**"GraphMem: Self-Evolving Graph-Based Memory for Production AI Agents"**

Key contributions:
- 99% token reduction through targeted graph retrieval
- 4.2× faster queries via O(1) entity indexing
- Self-evolution mechanisms (importance, decay, consolidation)
- Bounded memory growth (proven theorem)

Paper: [`paper/main.tex`](paper/main.tex)

## 📦 Dependencies

### Required
- Python 3.9+
- numpy
- pydantic
- openai

### Optional
- **Graph Storage**: neo4j
- **Caching**: redis
- **PDF**: PyMuPDF
- **Network**: networkx (for community detection)

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- Inspired by Microsoft GraphRAG and cognitive science research
- Built on Neo4j, Redis, and OpenAI

---

**Made with ❤️ by Al-Amin Ibrahim**

[![GitHub](https://img.shields.io/badge/GitHub-Al--aminI/GraphMem-blue)](https://github.com/Al-aminI/GraphMem)
[![PyPI](https://img.shields.io/badge/PyPI-agentic--graph--mem-green)](https://pypi.org/project/agentic-graph-mem/)

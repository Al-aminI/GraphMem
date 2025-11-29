# 🧠 GraphMem

**Production-Grade Agent Memory Framework for Agentic AI**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

GraphMem is a state-of-the-art, self-evolving graph-based memory system designed for production-scale agentic AI applications. It provides human-like memory capabilities with automatic consolidation, decay, and rehydration—all built on enterprise-grade storage backends.

## ✨ Key Features

### 🔄 Self-Evolving Memory
- **Memory Consolidation**: Automatically merges related memories into coherent knowledge
- **Importance Decay**: Less relevant memories naturally fade over time
- **Rehydration**: Revive and strengthen memories when accessed
- **Continuous Learning**: Memory improves through usage patterns

### 🕸️ Graph-Based Knowledge
- **Entity Resolution**: Intelligent deduplication and canonicalization
- **Community Detection**: Automatic topic clustering
- **Rich Relationships**: Capture complex entity connections
- **Semantic Search**: Find relevant context by meaning

### 📚 Multi-Modal Context Engineering
- **Text Documents**: Intelligent chunking with semantic boundaries
- **PDFs**: Extract text, images, and tables
- **Images**: OCR and vision model analysis
- **Audio**: Transcription to text
- **Web Pages**: Smart content extraction
- **Code Files**: Language-aware chunking
- **Structured Data**: JSON, CSV processing

### 🚀 Production Ready
- **Neo4j Backend**: Enterprise graph database
- **Redis Caching**: Sub-millisecond retrieval
- **Parallel Processing**: Concurrent knowledge extraction
- **Retry Logic**: Resilient to transient failures
- **Scalable**: Handles millions of memories

## 🏁 Quick Start

### Installation

```bash
pip install agentic-graph-mem
```

Or with all dependencies:

```bash
pip install agentic-graph-mem[all]
```

### Basic Usage

```python
from graphmem import GraphMem

# Initialize with sensible defaults
memory = GraphMem()

# Ingest information
memory.ingest("""
TechCorp announced today that John Smith has been appointed as their new CEO.
Smith brings 20 years of experience from leading AI companies including DeepMind
and OpenAI. The company's stock rose 15% on the news.
""")

# Query the memory
response = memory.query("Who is the new CEO of TechCorp?")
print(response.answer)
# Output: "John Smith has been appointed as the new CEO of TechCorp."

# Memory evolves automatically
memory.evolve()
```

### Configuration

```python
from graphmem import GraphMem, MemoryConfig

config = MemoryConfig(
    # LLM settings
    llm_provider="azure_openai",
    llm_api_key="your-api-key",
    llm_endpoint="https://your-endpoint.openai.azure.com",
    llm_deployment="gpt-4o",
    
    # Embedding settings
    embedding_provider="azure_openai",
    embedding_deployment="text-embedding-3-small",
    
    # Storage settings
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    
    # Cache settings
    redis_url="redis://localhost:6379",
    
    # Evolution settings
    auto_evolve=True,
    evolution_interval=3600,  # seconds
    consolidation_threshold=0.85,
    decay_rate=0.01,
)

memory = GraphMem(config)
```

## 🏗️ Architecture

```
GraphMem
├── Core
│   ├── GraphMem          # Main interface
│   ├── Memory            # Memory unit (nodes, edges, clusters)
│   ├── MemoryNode        # Entity representation
│   ├── MemoryEdge        # Relationship representation
│   └── MemoryCluster     # Community/topic grouping
│
├── Graph
│   ├── KnowledgeGraph    # Knowledge extraction & storage
│   ├── EntityResolver    # Entity deduplication
│   └── CommunityDetector # Topic clustering
│
├── Evolution
│   ├── MemoryEvolution   # Evolution orchestrator
│   ├── MemoryDecay       # Importance decay
│   ├── Consolidation     # Memory merging
│   └── Rehydration       # Memory restoration
│
├── Retrieval
│   ├── QueryEngine       # Query processing
│   ├── MemoryRetriever   # Context retrieval
│   └── SemanticSearch    # Embedding search
│
├── Context
│   ├── ContextEngine     # Context window construction
│   ├── DocumentChunker   # Semantic chunking
│   └── MultiModalProcessor # Multi-modal handling
│
├── LLM
│   ├── LLMProvider       # LLM abstraction
│   └── EmbeddingProvider # Embedding abstraction
│
└── Stores
    ├── Neo4jStore        # Graph persistence
    └── RedisCache        # Caching layer
```

## 📖 Advanced Usage

### Custom Knowledge Extraction

```python
from graphmem import GraphMem, KnowledgeGraph

# Create with custom extraction prompt
memory = GraphMem(
    extraction_prompt="""
    Extract entities and relationships from the text.
    Focus on: People, Organizations, Products, Events
    
    Text: {text}
    """
)
```

### Multi-Modal Ingestion

```python
from graphmem import GraphMem

memory = GraphMem()

# Ingest PDF
memory.ingest_file("report.pdf", modality="pdf")

# Ingest image
memory.ingest_file("diagram.png", modality="image")

# Ingest audio (transcribes automatically)
memory.ingest_file("meeting.mp3", modality="audio")

# Ingest web page
memory.ingest_url("https://example.com/article")

# Ingest code
memory.ingest_file("main.py", modality="code")
```

### Manual Evolution Control

```python
from graphmem import GraphMem, MemoryConfig

config = MemoryConfig(auto_evolve=False)
memory = GraphMem(config)

# Add content
memory.ingest("...")

# Manually trigger evolution
memory.consolidate()  # Merge similar memories
memory.decay()        # Apply importance decay
memory.prune()        # Remove low-importance memories
```

### Query with Filters

```python
response = memory.query(
    "What happened at TechCorp?",
    filters={
        "entity_type": "Organization",
        "min_importance": 5,
    },
    top_k=20,
    include_context=True,
)

print(response.answer)
print(response.confidence)
print(response.context)
```

### Direct Graph Access

```python
from graphmem import GraphMem

memory = GraphMem()

# Get entities
entities = memory.get_entities(entity_type="Person")

# Get relationships
relationships = memory.get_relationships(
    source="John Smith",
    relation_type="CEO_OF"
)

# Get communities
communities = memory.get_communities()
```

## 🔧 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm_provider` | LLM provider (azure_openai, openai, anthropic, ollama) | `azure_openai` |
| `embedding_provider` | Embedding provider | `azure_openai` |
| `neo4j_uri` | Neo4j connection URI | `bolt://localhost:7687` |
| `redis_url` | Redis connection URL | `redis://localhost:6379` |
| `auto_evolve` | Enable automatic memory evolution | `True` |
| `evolution_interval` | Seconds between evolution cycles | `3600` |
| `consolidation_threshold` | Similarity threshold for merging | `0.85` |
| `decay_rate` | Daily decay rate for importance | `0.01` |
| `chunk_size` | Document chunk size in characters | `1000` |
| `chunk_overlap` | Chunk overlap in characters | `200` |
| `top_k` | Default number of retrieval results | `10` |
| `min_similarity` | Minimum similarity for retrieval | `0.5` |

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=graphmem

# Run specific test
pytest tests/test_memory.py::test_ingestion
```

## 📦 Dependencies

### Required
- Python 3.9+
- numpy
- pydantic

### Optional (by feature)
- **LLM**: openai, anthropic
- **Storage**: neo4j, redis
- **PDF**: PyMuPDF or PyPDF2
- **OCR**: pytesseract, Pillow
- **Audio**: openai-whisper
- **Web**: beautifulsoup4, requests
- **Local Embeddings**: sentence-transformers

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with inspiration from:
- GraphRAG by Microsoft
- LlamaIndex
- Human cognitive science research on memory consolidation

---

**Made with ❤️ by Ameer AI**


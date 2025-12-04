"""
GraphMem - Production-Grade Agent Memory Framework

A state-of-the-art, self-evolving graph-based memory system for agentic AI applications.
Designed for production scale with enterprise-grade reliability.

Features:
- Self-improving and self-evolving memory like humans
- Graph-based knowledge representation with Neo4j
- Multi-modal context engineering
- Semantic search with embeddings
- Memory consolidation, decay, and rehydration
- Production-ready with Redis caching
- Simple API with powerful capabilities

Example Usage:
    >>> from graphmem import GraphMem
    >>> 
    >>> # Initialize with minimal configuration
    >>> memory = GraphMem()
    >>> 
    >>> # Ingest documents and build knowledge graph
    >>> memory.ingest("The CEO of TechCorp, John Smith, announced...")
    >>> 
    >>> # Query the memory
    >>> response = memory.query("Who is the CEO of TechCorp?")
    >>> 
    >>> # Memory evolves and consolidates automatically
    >>> memory.evolve()

Author: Ameer AI
Version: 1.0.0
License: MIT
"""

from graphmem.version import __version__, __version_info__
from graphmem.core.memory import GraphMem, MemoryConfig
from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryQuery,
    MemoryResponse,
    MemoryImportance,
    MemoryState,
    EvolutionType,
    EvolutionEvent,
)
from graphmem.core.exceptions import (
    GraphMemError,
    IngestionError,
    QueryError,
    StorageError,
    EvolutionError,
    ConfigurationError,
)

# Graph components
from graphmem.graph.knowledge_graph import KnowledgeGraph
from graphmem.graph.entity_resolver import EntityResolver
from graphmem.graph.community_detector import CommunityDetector

# Evolution components  
from graphmem.evolution.memory_evolution import MemoryEvolution
from graphmem.evolution.consolidation import MemoryConsolidation
from graphmem.evolution.decay import MemoryDecay
from graphmem.evolution.rehydration import GraphRehydration

# Retrieval components
from graphmem.retrieval.retriever import MemoryRetriever
from graphmem.retrieval.query_engine import QueryEngine
from graphmem.retrieval.semantic_search import SemanticSearch

# Context engineering
from graphmem.context.context_engine import ContextEngine
from graphmem.context.chunker import DocumentChunker
from graphmem.context.multimodal import MultiModalProcessor
from graphmem.context.extractors import (
    extract_webpage,
    check_webpage_url,
)

# LLM providers
from graphmem.llm.providers import LLMProvider
from graphmem.llm.embeddings import EmbeddingProvider

# Storage backends
from graphmem.stores.neo4j_store import Neo4jStore
from graphmem.stores.redis_cache import RedisCache
from graphmem.stores.memory_store import InMemoryStore, InMemoryCache

# Optional Turso support (SQLite-based, offline-first)
try:
    from graphmem.stores.turso_store import TursoStore, TursoCache
except ImportError:
    TursoStore = None
    TursoCache = None

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Core
    "GraphMem",
    "MemoryConfig",
    "Memory",
    "MemoryNode",
    "MemoryEdge",
    "MemoryCluster",
    "MemoryQuery",
    "MemoryResponse",
    "MemoryImportance",
    "MemoryState",
    "EvolutionType",
    "EvolutionEvent",
    # Exceptions
    "GraphMemError",
    "IngestionError",
    "QueryError",
    "StorageError",
    "EvolutionError",
    "ConfigurationError",
    # Graph
    "KnowledgeGraph",
    "EntityResolver",
    "CommunityDetector",
    # Evolution
    "MemoryEvolution",
    "MemoryConsolidation",
    "MemoryDecay",
    "GraphRehydration",
    # Retrieval
    "MemoryRetriever",
    "QueryEngine",
    "SemanticSearch",
    # Context
    "ContextEngine",
    "DocumentChunker",
    "MultiModalProcessor",
    # Extractors (text/webpage only)
    "extract_webpage",
    "check_webpage_url",
    # LLM
    "LLMProvider",
    "EmbeddingProvider",
    # Storage
    "Neo4jStore",
    "RedisCache",
    "InMemoryStore",
    "InMemoryCache",
    "TursoStore",
    "TursoCache",
]

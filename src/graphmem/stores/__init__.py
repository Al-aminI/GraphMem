"""
GraphMem Storage Module

Production-grade storage backends for persistent memory.

Available backends:
- Neo4jStore: Full graph database with native graph algorithms
- TursoStore: SQLite-based, offline-first, lightweight alternative
- InMemoryStore: Simple in-memory storage (default)
- RedisCache: High-performance distributed caching
- TursoCache: SQLite-based persistent caching
"""

from graphmem.stores.neo4j_store import Neo4jStore
from graphmem.stores.redis_cache import RedisCache
from graphmem.stores.memory_store import InMemoryStore, InMemoryCache

# Optional Turso support
try:
    from graphmem.stores.turso_store import TursoStore, TursoCache, TURSO_AVAILABLE
except ImportError:
    TursoStore = None
    TursoCache = None
    TURSO_AVAILABLE = False

__all__ = [
    "Neo4jStore",
    "RedisCache",
    "InMemoryStore",
    "InMemoryCache",
    "TursoStore",
    "TursoCache",
    "TURSO_AVAILABLE",
]


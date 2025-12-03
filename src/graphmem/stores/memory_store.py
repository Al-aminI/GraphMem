"""
GraphMem In-Memory Store

Simple in-memory storage backend for development and single-node deployments.
No external dependencies required.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, List
from copy import deepcopy

from graphmem.core.memory_types import Memory, MemoryNode, MemoryEdge, MemoryCluster

logger = logging.getLogger(__name__)


class InMemoryStore:
    """
    In-memory storage backend for GraphMem.
    
    Perfect for:
    - Development and testing
    - Single-node deployments
    - Applications that don't need persistence
    - Quick prototyping
    
    Note: Data is lost when the process ends. Use Neo4jStore for persistence.
    """
    
    def __init__(self):
        """Initialize in-memory store."""
        self._memories: Dict[str, Memory] = {}
        logger.info("InMemoryStore initialized")
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to in-memory storage."""
        self._memories[memory.id] = deepcopy(memory)
        logger.debug(f"Saved memory {memory.id} to in-memory store")
    
    def load_memory(self, memory_id: str) -> Optional[Memory]:
        """Load memory from in-memory storage."""
        memory = self._memories.get(memory_id)
        if memory:
            return deepcopy(memory)
        return None
    
    def delete_memory(self, memory_id: str) -> None:
        """Delete memory from storage."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            logger.debug(f"Deleted memory {memory_id}")
    
    def clear_memory(self, memory_id: str) -> None:
        """Clear all data in a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
    
    def list_memories(self) -> List[str]:
        """List all memory IDs."""
        return list(self._memories.keys())
    
    def close(self) -> None:
        """Close the store (no-op for in-memory)."""
        pass
    
    def health_check(self) -> bool:
        """Check if store is healthy."""
        return True


class InMemoryCache:
    """
    Simple in-memory cache (replacement for Redis when not available).
    """
    
    def __init__(self, ttl: int = 3600):
        """Initialize in-memory cache."""
        self._cache: Dict[str, any] = {}
        self.ttl = ttl
        logger.info("InMemoryCache initialized")
    
    def get(self, key: str) -> Optional[any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def invalidate(self, prefix: str) -> None:
        """Invalidate all keys with prefix."""
        keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]
    
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self._cache.get(f"emb:{key}")
    
    def cache_embedding(self, key: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        self._cache[f"emb:{key}"] = embedding
    
    def close(self) -> None:
        """Close cache (no-op for in-memory)."""
        pass


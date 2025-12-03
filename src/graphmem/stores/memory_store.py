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
    
    def load_memory(self, memory_id: str, user_id: str = None) -> Optional[Memory]:
        """
        Load memory from in-memory storage.
        
        Args:
            memory_id: Memory ID to load
            user_id: User ID for multi-tenant isolation (filters nodes)
        """
        memory = self._memories.get(memory_id)
        if memory:
            loaded = deepcopy(memory)
            # Filter by user_id if provided
            if user_id:
                loaded.nodes = {
                    nid: node for nid, node in loaded.nodes.items()
                    if node.user_id == user_id or node.user_id is None
                }
                # Filter edges to only include those between remaining nodes
                loaded.edges = {
                    eid: edge for eid, edge in loaded.edges.items()
                    if edge.source_id in loaded.nodes and edge.target_id in loaded.nodes
                }
            return loaded
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
    
    Provides the same interface as RedisCache for seamless fallback.
    Includes multi-tenant support via user_id in cache keys.
    """
    
    def __init__(self, ttl: int = 3600):
        """Initialize in-memory cache."""
        self._cache: Dict[str, any] = {}
        self.ttl = ttl
        logger.info("InMemoryCache initialized")
    
    def _key(self, *parts: str) -> str:
        """Build a cache key."""
        return ":".join(parts)
    
    def get(self, *key_parts: str) -> Optional[any]:
        """Get value from cache."""
        key = self._key(*key_parts) if len(key_parts) > 1 else key_parts[0]
        return self._cache.get(key)
    
    def set(self, *key_parts: str, value: any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        # Last part before value is the key
        key = self._key(*key_parts)
        self._cache[key] = value
        return True
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def invalidate(self, memory_id: str, user_id: str = "default") -> None:
        """
        Invalidate all cache entries for a user's memory.
        
        Multi-tenant safe: matches user_id in key pattern.
        """
        pattern_parts = [user_id, memory_id]
        keys_to_delete = [
            k for k in self._cache 
            if all(part in k for part in pattern_parts)
        ]
        for key in keys_to_delete:
            del self._cache[key]
        logger.debug(f"Invalidated {len(keys_to_delete)} in-memory cache entries")
    
    # Specialized cache methods matching RedisCache interface
    
    def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self._cache.get(f"embedding:{text_hash}")
    
    def cache_embedding(self, text_hash: str, embedding: List[float], ttl: int = 86400) -> bool:
        """Cache an embedding."""
        self._cache[f"embedding:{text_hash}"] = embedding
        return True
    
    def get_search_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[list]:
        """Get cached search results."""
        key = self._key("search", user_id, memory_id, query_hash)
        return self._cache.get(key)
    
    def cache_search_result(
        self,
        memory_id: str,
        query_hash: str,
        results: list,
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """Cache search results."""
        key = self._key("search", user_id, memory_id, query_hash)
        self._cache[key] = results
        return True
    
    def get_query_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = self._key("query", user_id, memory_id, query_hash)
        return self._cache.get(key)
    
    def cache_query_result(
        self,
        memory_id: str,
        query_hash: str,
        result: Dict[str, Any],
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """Cache query result."""
        key = self._key("query", user_id, memory_id, query_hash)
        self._cache[key] = result
        return True
    
    def get_community_context(
        self,
        memory_id: str,
        community_id: int,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached community context."""
        key = self._key("community", user_id, memory_id, str(community_id))
        return self._cache.get(key)
    
    def cache_community_context(
        self,
        memory_id: str,
        community_id: int,
        context: Dict[str, Any],
        user_id: str = "default",
    ) -> bool:
        """Cache community context."""
        key = self._key("community", user_id, memory_id, str(community_id))
        self._cache[key] = context
        return True
    
    def close(self) -> None:
        """Close cache (no-op for in-memory)."""
        pass


"""
GraphMem Redis Cache

High-performance caching layer for memory operations.
Dramatically reduces latency for frequently accessed data.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis caching layer for GraphMem.
    
    Caches:
    - Memory state (entity_info, community summaries)
    - Query results
    - Community context
    - Embeddings
    """
    
    def __init__(
        self,
        url: str,
        ttl: int = 3600,
        prefix: str = "graphmem",
    ):
        """
        Initialize Redis cache.
        
        Args:
            url: Redis connection URL
            ttl: Default TTL in seconds
            prefix: Key prefix for all GraphMem keys
        """
        self.url = url
        self.ttl = ttl
        self.prefix = prefix
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(self.url)
            except ImportError:
                raise ImportError("redis package required: pip install redis")
        return self._client
    
    def _key(self, *parts: str) -> str:
        """Build a cache key."""
        return f"{self.prefix}:{':'.join(parts)}"
    
    def get(self, *key_parts: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            key = self._key(*key_parts)
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None
    
    def set(
        self,
        *key_parts: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in cache."""
        try:
            key = self._key(*key_parts)
            serialized = json.dumps(value, default=str)
            self.client.setex(key, ttl or self.ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False
    
    def delete(self, *key_parts: str) -> bool:
        """Delete a value from cache."""
        try:
            key = self._key(*key_parts)
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return False
    
    def invalidate(self, memory_id: str, user_id: str = "default") -> None:
        """
        Invalidate all cache entries for a user's memory.
        
        Multi-tenant safe: uses user_id in the pattern.
        """
        try:
            # Invalidate search, query, and community caches for this user's memory
            patterns = [
                self._key("search", user_id, memory_id, "*"),
                self._key("query", user_id, memory_id, "*"),
                self._key("community", user_id, memory_id, "*"),
                self._key("state", user_id, memory_id),
            ]
            
            total_deleted = 0
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.client.delete(*keys)
                        total_deleted += len(keys)
                    if cursor == 0:
                        break
            
            logger.debug(f"Invalidated {total_deleted} cache entries for user={user_id}, memory={memory_id}")
        except Exception as e:
            logger.warning(f"Redis invalidate failed: {e}")
    
    # Specialized cache methods with multi-tenant support
    
    def cache_memory_state(
        self, 
        memory_id: str, 
        state: Dict[str, Any],
        user_id: str = "default",
    ) -> bool:
        """Cache memory state (entity_info, community_summary)."""
        return self.set("state", user_id, memory_id, value=state)
    
    def get_memory_state(
        self, 
        memory_id: str,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached memory state."""
        return self.get("state", user_id, memory_id)
    
    def cache_query_result(
        self,
        memory_id: str,
        query_hash: str,
        result: Dict[str, Any],
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """
        Cache a query result (short TTL, 5 min default).
        
        Multi-tenant safe: cache key includes user_id.
        """
        return self.set("query", user_id, memory_id, query_hash, value=result, ttl=ttl)
    
    def get_query_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result (multi-tenant safe)."""
        return self.get("query", user_id, memory_id, query_hash)
    
    def cache_search_result(
        self,
        memory_id: str,
        query_hash: str,
        results: list,
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """
        Cache semantic search results (short TTL, 5 min default).
        
        Multi-tenant safe: cache key includes user_id.
        """
        return self.set("search", user_id, memory_id, query_hash, value=results, ttl=ttl)
    
    def get_search_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[list]:
        """Get cached search results (multi-tenant safe)."""
        return self.get("search", user_id, memory_id, query_hash)
    
    def cache_community_context(
        self,
        memory_id: str,
        community_id: int,
        context: Dict[str, Any],
        user_id: str = "default",
    ) -> bool:
        """Cache community context (multi-tenant safe)."""
        return self.set("community", user_id, memory_id, str(community_id), value=context)
    
    def get_community_context(
        self,
        memory_id: str,
        community_id: int,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached community context (multi-tenant safe)."""
        return self.get("community", user_id, memory_id, str(community_id))
    
    def cache_embedding(
        self,
        text_hash: str,
        embedding: list,
        ttl: int = 86400,  # 24 hours
    ) -> bool:
        """Cache an embedding (shared across users for efficiency)."""
        return self.set("embedding", text_hash, value=embedding, ttl=ttl)
    
    def get_embedding(self, text_hash: str) -> Optional[list]:
        """Get cached embedding."""
        return self.get("embedding", text_hash)
    
    def invalidate_user(self, user_id: str, memory_id: str) -> None:
        """Invalidate all cache entries for a user's memory."""
        try:
            pattern = self._key("*", user_id, memory_id, "*")
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
            logger.debug(f"Invalidated cache for user {user_id}, memory {memory_id}")
        except Exception as e:
            logger.warning(f"Redis invalidate_user failed: {e}")
    
    def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

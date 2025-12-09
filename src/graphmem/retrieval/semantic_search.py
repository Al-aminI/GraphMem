"""
GraphMem Semantic Search

High-performance semantic search over memory nodes.
Supports both in-memory search and Neo4j vector index for scalability.
"""

from __future__ import annotations
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np

from graphmem.core.memory_types import MemoryNode, MemoryEdge


def _get_importance_value(importance) -> float:
    """Safely get numeric value from importance (enum or float)."""
    if hasattr(importance, 'value'):
        return importance.value
    return float(importance) if importance is not None else 0.5


if TYPE_CHECKING:
    from graphmem.stores.neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic search engine for memory retrieval.
    
    Features:
    - Embedding-based similarity search
    - Neo4j vector index support for large-scale search
    - Hybrid search (semantic + keyword)
    - Evolution-aware ranking (importance, recency, access count)
    - Filtered search with metadata
    - Caching for performance
    """
    
    def __init__(
        self,
        embeddings,
        cache=None,
        top_k: int = 10,
        min_similarity: float = 0.5,
        neo4j_store: Optional["Neo4jStore"] = None,
        memory_id: Optional[str] = None,
        user_id: str = "default",
    ):
        """
        Initialize semantic search.
        
        Args:
            embeddings: Embedding provider
            cache: Optional cache for embeddings
            top_k: Default number of results
            min_similarity: Minimum similarity threshold
            neo4j_store: Optional Neo4j store for vector index search
            memory_id: Memory ID for Neo4j vector search
            user_id: User ID for multi-tenant isolation
        """
        self.embeddings = embeddings
        self.cache = cache
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.neo4j_store = neo4j_store
        self.memory_id = memory_id
        self.user_id = user_id  # Multi-tenant isolation
        
        # In-memory index for fast search (fallback when Neo4j not available)
        self._index: Dict[str, np.ndarray] = {}
        self._node_lookup: Dict[str, MemoryNode] = {}
        
        # Track if Neo4j vector search is available
        self._use_neo4j_vector = False
        if neo4j_store and memory_id:
            try:
                self._use_neo4j_vector = neo4j_store.use_vector_index
                if self._use_neo4j_vector:
                    neo4j_store.ensure_vector_index(memory_id)
                    logger.info("Neo4j vector index enabled for semantic search")
            except Exception as e:
                logger.warning(f"Neo4j vector index not available: {e}")
    
    def index_nodes(self, nodes: List[MemoryNode]) -> None:
        """
        Index nodes for search.
        
        Args:
            nodes: Nodes to index
        """
        for node in nodes:
            text = node.description or node.name
            
            try:
                embedding = self.embeddings.embed_text(text)
                if embedding:
                    self._index[node.id] = np.array(embedding)
                    self._node_lookup[node.id] = node
            except Exception as e:
                logger.warning(f"Failed to index node {node.id}: {e}")
        
        logger.debug(f"Indexed {len(self._index)} nodes")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Search for similar nodes.
        
        Uses Neo4j vector index if available, otherwise falls back to in-memory search.
        Redis caching accelerates repeated queries.
        
        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity
            filters: Optional filters
        
        Returns:
            List of (node, similarity_score) tuples
        """
        top_k = top_k or self.top_k
        min_similarity = min_similarity or self.min_similarity
        
        # Generate query hash for caching
        query_hash = hashlib.md5(
            f"{query}:{top_k}:{min_similarity}:{json.dumps(filters or {}, sort_keys=True)}".encode()
        ).hexdigest()
        
        # Check Redis cache first (multi-tenant safe)
        if self.cache and self.memory_id:
            try:
                cached = self.cache.get_search_result(
                    memory_id=self.memory_id,
                    query_hash=query_hash,
                    user_id=self.user_id,
                )
                if cached:
                    logger.debug(f"Cache hit for search query: {query[:50]}")
                    # Reconstruct MemoryNode objects from cached data
                    return self._deserialize_results(cached)
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        # Get query embedding
        try:
            query_embedding = self.embeddings.embed_text(query)
            if not query_embedding:
                return []
            query_vector = np.array(query_embedding)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
        
        # Use Neo4j vector search if available (faster for large datasets)
        # Includes multi-tenant isolation via user_id
        if self._use_neo4j_vector and self.neo4j_store and self.memory_id:
            try:
                results = self.neo4j_store.vector_search(
                    memory_id=self.memory_id,
                    query_embedding=list(query_embedding),
                    top_k=top_k,
                    min_score=min_similarity,
                    user_id=self.user_id,  # Multi-tenant isolation
                )
                
                # Apply filters
                if filters:
                    results = [(n, s) for n, s in results if self._matches_filters(n, filters)]
                
                # Cache the results
                self._cache_results(query_hash, results)
                
                logger.debug(f"Neo4j vector search returned {len(results)} results")
                return results
                
            except Exception as e:
                logger.warning(f"Neo4j vector search failed, falling back to in-memory: {e}")
        
        # Fallback: In-memory search with HYBRID ALIAS MATCHING
        if not self._index:
            return []
        
        # Pre-process query for alias matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Calculate similarities with importance weighting
        # This is where evolution features (decay, consolidation) actually matter!
        results = []
        for node_id, node_vector in self._index.items():
            similarity = self._cosine_similarity(query_vector, node_vector)
            
            # CRITICAL: Also check for ALIAS MATCHES (hybrid search)
            # This catches queries like "What did Dr. Chen do?" when node has alias "Dr. Chen"
            alias_boost = 0.0
            node = self._node_lookup.get(node_id)
            if node and hasattr(node, 'aliases') and node.aliases:
                for alias in node.aliases:
                    alias_lower = alias.lower()
                    # Exact alias match in query
                    if alias_lower in query_lower:
                        alias_boost = max(alias_boost, 0.3)  # Strong boost
                        logger.debug(f"Alias match: '{alias}' in query '{query[:50]}'")
                        break
                    # Partial word match
                    alias_words = set(alias_lower.split())
                    if alias_words & query_words:
                        alias_boost = max(alias_boost, 0.15)  # Medium boost
                
                # Also check canonical name
                if node.canonical_name and node.canonical_name.lower() in query_lower:
                    alias_boost = max(alias_boost, 0.25)
                
                # Also check node name
                if node.name.lower() in query_lower:
                    alias_boost = max(alias_boost, 0.25)
            
            # Adjust effective similarity with alias boost
            effective_similarity = min(1.0, similarity + alias_boost)
            
            if effective_similarity >= min_similarity:
                if node:
                    # Apply filters
                    if filters and not self._matches_filters(node, filters):
                        continue
                    
                    # Skip nodes that have decayed too much (EPHEMERAL = 0)
                    # Evolution's decay feature marks unimportant nodes
                    importance_val = _get_importance_value(node.importance)
                    if importance_val == 0:  # EPHEMERAL - skip decayed nodes
                        continue
                    
                    # Weight similarity by importance (evolution matters!)
                    # importance.value: CRITICAL=10, VERY_HIGH=8, HIGH=6, MEDIUM=5, LOW=3, VERY_LOW=1
                    importance_weight = importance_val / 10.0  # Normalize to 0-1
                    
                    # Recency boost - recently accessed nodes are more relevant
                    recency_boost = 0.0
                    if node.accessed_at:
                        from datetime import datetime
                        hours_since_access = (datetime.utcnow() - node.accessed_at).total_seconds() / 3600
                        if hours_since_access < 24:
                            recency_boost = 0.1 * (1 - hours_since_access / 24)  # Up to 10% boost
                    
                    # Access count boost - frequently accessed nodes matter more
                    access_boost = min(0.1, node.access_count * 0.01)  # Up to 10% boost
                    
                    # Combined score: 50% similarity + 25% importance + 15% alias + 5% recency + 5% access
                    combined_score = (
                        0.50 * similarity + 
                        0.25 * importance_weight + 
                        0.15 * (alias_boost * 3.33) +  # Normalize alias boost to 0-0.5 range
                        recency_boost + 
                        access_boost
                    )
                    
                    results.append((node, combined_score))
        
        # Sort by combined score (similarity + importance + recency + access) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:top_k]
        
        # Cache the results
        self._cache_results(query_hash, final_results)
        
        return final_results
    
    def _cache_results(self, query_hash: str, results: List[Tuple[MemoryNode, float]]) -> None:
        """Cache search results in Redis (multi-tenant safe)."""
        if self.cache and self.memory_id and results:
            try:
                serialized = [
                    {
                        "node": self._serialize_node(node),
                        "score": score,
                    }
                    for node, score in results
                ]
                self.cache.cache_search_result(
                    memory_id=self.memory_id,
                    query_hash=query_hash,
                    results=serialized,
                    user_id=self.user_id,
                )
                logger.debug(f"Cached {len(results)} search results")
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")
    
    def _serialize_node(self, node: MemoryNode) -> Dict[str, Any]:
        """Serialize a MemoryNode for caching."""
        return {
            "id": node.id,
            "name": node.name,
            "entity_type": node.entity_type,
            "description": node.description,
            "canonical_name": node.canonical_name,
            "aliases": list(node.aliases),
            "properties": node.properties,
            "importance": node.importance.name,
            "access_count": node.access_count,
            "user_id": node.user_id,
            "memory_id": node.memory_id,
            "created_at": str(node.created_at) if node.created_at else None,
            "accessed_at": str(node.accessed_at) if node.accessed_at else None,
        }
    
    def _deserialize_results(self, cached: List[Dict]) -> List[Tuple[MemoryNode, float]]:
        """Deserialize cached search results."""
        from graphmem.core.memory_types import MemoryImportance
        from datetime import datetime
        
        results = []
        for item in cached:
            node_data = item["node"]
            score = item["score"]
            
            # Parse datetime strings
            created_at = None
            if node_data.get("created_at") and node_data["created_at"] != "None":
                try:
                    created_at = datetime.fromisoformat(node_data["created_at"].replace("Z", "+00:00"))
                except:
                    pass
            
            accessed_at = None
            if node_data.get("accessed_at") and node_data["accessed_at"] != "None":
                try:
                    accessed_at = datetime.fromisoformat(node_data["accessed_at"].replace("Z", "+00:00"))
                except:
                    pass
            
            node = MemoryNode(
                id=node_data["id"],
                name=node_data["name"],
                entity_type=node_data.get("entity_type", "Entity"),
                description=node_data.get("description"),
                canonical_name=node_data.get("canonical_name"),
                aliases=set(node_data.get("aliases", [])),
                properties=node_data.get("properties", {}),
                importance=MemoryImportance[node_data.get("importance", "MEDIUM")],
                access_count=node_data.get("access_count", 0),
                user_id=node_data.get("user_id"),
                memory_id=node_data.get("memory_id"),
                created_at=created_at,
                accessed_at=accessed_at,
            )
            results.append((node, score))
        
        return results
    
    def enable_neo4j_vector(self, neo4j_store: "Neo4jStore", memory_id: str) -> bool:
        """
        Enable Neo4j vector search.
        
        Args:
            neo4j_store: Neo4j store instance
            memory_id: Memory ID
            
        Returns:
            True if Neo4j vector search was enabled
        """
        self.neo4j_store = neo4j_store
        self.memory_id = memory_id
        
        try:
            if neo4j_store.use_vector_index:
                neo4j_store.ensure_vector_index(memory_id)
                self._use_neo4j_vector = neo4j_store.has_vector_support()
                if self._use_neo4j_vector:
                    logger.info("Neo4j vector search enabled")
                return self._use_neo4j_vector
        except Exception as e:
            logger.warning(f"Could not enable Neo4j vector search: {e}")
        
        return False
    
    @property
    def using_neo4j_vector(self) -> bool:
        """Check if Neo4j vector search is being used."""
        return self._use_neo4j_vector
    
    def find_similar_entities(
        self,
        query: str,
        entity_names: List[str],
        top_k: int = 5,
    ) -> List[str]:
        """
        Find entity names similar to query.
        
        Args:
            query: Query text
            entity_names: List of entity names to search
            top_k: Number of results
        
        Returns:
            List of matching entity names
        """
        if not entity_names:
            return []
        
        try:
            query_embedding = self.embeddings.embed_text(query)
            entity_embeddings = self.embeddings.embed_batch(entity_names)
            
            if not query_embedding or not entity_embeddings:
                return []
            
            query_vector = np.array(query_embedding)
            
            scored = []
            for i, name in enumerate(entity_names):
                entity_vector = np.array(entity_embeddings[i])
                similarity = self._cosine_similarity(query_vector, entity_vector)
                scored.append((name, similarity))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return [name for name, _ in scored[:top_k]]
            
        except Exception as e:
            logger.error(f"Entity similarity search failed: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _matches_filters(self, node: MemoryNode, filters: Dict[str, Any]) -> bool:
        """Check if node matches filters."""
        for key, value in filters.items():
            if key == "entity_type":
                if node.entity_type.lower() != value.lower():
                    return False
            elif key == "min_importance":
                if _get_importance_value(node.importance) < value:
                    return False
            elif key == "state":
                if node.state.name != value:
                    return False
            elif key in node.properties:
                if node.properties[key] != value:
                    return False
        return True
    
    def clear_index(self) -> None:
        """Clear the search index."""
        self._index.clear()
        self._node_lookup.clear()


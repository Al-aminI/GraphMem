"""
GraphMem Semantic Search

High-performance semantic search over memory nodes.
Uses embeddings for similarity matching.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from graphmem.core.memory_types import MemoryNode, MemoryEdge

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic search engine for memory retrieval.
    
    Features:
    - Embedding-based similarity search
    - Hybrid search (semantic + keyword)
    - Filtered search with metadata
    - Caching for performance
    """
    
    def __init__(
        self,
        embeddings,
        cache=None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ):
        """
        Initialize semantic search.
        
        Args:
            embeddings: Embedding provider
            cache: Optional cache for embeddings
            top_k: Default number of results
            min_similarity: Minimum similarity threshold
        """
        self.embeddings = embeddings
        self.cache = cache
        self.top_k = top_k
        self.min_similarity = min_similarity
        
        # In-memory index for fast search
        self._index: Dict[str, np.ndarray] = {}
        self._node_lookup: Dict[str, MemoryNode] = {}
    
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
        
        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity
            filters: Optional filters
        
        Returns:
            List of (node, similarity_score) tuples
        """
        if not self._index:
            return []
        
        top_k = top_k or self.top_k
        min_similarity = min_similarity or self.min_similarity
        
        # Get query embedding
        try:
            query_embedding = self.embeddings.embed_text(query)
            if not query_embedding:
                return []
            query_vector = np.array(query_embedding)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
        
        # Calculate similarities
        results = []
        for node_id, node_vector in self._index.items():
            similarity = self._cosine_similarity(query_vector, node_vector)
            
            if similarity >= min_similarity:
                node = self._node_lookup.get(node_id)
                if node:
                    # Apply filters
                    if filters and not self._matches_filters(node, filters):
                        continue
                    results.append((node, similarity))
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
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
                if node.importance.value < value:
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


"""
GraphMem Memory Retriever

Retrieves relevant memories using multiple strategies:
- Semantic search
- Graph traversal
- Community-based retrieval
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryQuery,
)
from graphmem.retrieval.semantic_search import SemanticSearch

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Retrieves relevant memories for a query.
    
    Combines multiple retrieval strategies:
    1. Semantic search - find nodes by meaning
    2. Graph traversal - expand to related nodes
    3. Community retrieval - get cluster summaries
    """
    
    def __init__(
        self,
        embeddings,
        store,
        cache=None,
        top_k: int = 10,
        min_similarity: float = 0.5,
        memory_id: Optional[str] = None,
        user_id: str = "default",
    ):
        """
        Initialize retriever.
        
        Args:
            embeddings: Embedding provider
            store: Graph store (Neo4jStore for vector search, or InMemoryStore)
            cache: Optional cache
            top_k: Default number of results
            min_similarity: Minimum similarity threshold
            memory_id: Memory ID (used for Neo4j vector search)
            user_id: User ID for multi-tenant isolation
        """
        self.embeddings = embeddings
        self.store = store
        self.cache = cache
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.memory_id = memory_id
        self.user_id = user_id
        
        # Check if store is Neo4j for vector search
        neo4j_store = None
        if hasattr(store, 'vector_search') and hasattr(store, 'use_vector_index'):
            neo4j_store = store
        
        self.semantic_search = SemanticSearch(
            embeddings=embeddings,
            cache=cache,
            top_k=top_k,
            min_similarity=min_similarity,
            neo4j_store=neo4j_store,
            memory_id=memory_id,
            user_id=user_id,  # Multi-tenant isolation
        )
    
    def retrieve(
        self,
        query: MemoryQuery,
        memory: Memory,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for a query.
        
        Args:
            query: Query specification
            memory: Memory to search
        
        Returns:
            Dict with nodes, edges, clusters, and context
        """
        # Index memory for search
        self.semantic_search.index_nodes(list(memory.nodes.values()))
        
        # Semantic search for relevant nodes
        node_results = self.semantic_search.search(
            query=query.query,
            top_k=query.top_k,
            min_similarity=query.min_similarity,
            filters=query.filters,
        )
        
        nodes = [node for node, _ in node_results]
        scores = {node.id: score for node, score in node_results}
        
        # Expand via graph traversal
        if nodes:
            expanded_nodes, edges = self._expand_graph(
                initial_nodes=nodes,
                memory=memory,
                max_hops=1,
            )
            
            # Add expanded nodes (with lower scores)
            for node in expanded_nodes:
                if node.id not in scores:
                    nodes.append(node)
                    scores[node.id] = 0.3  # Lower score for expanded
        else:
            edges = []
        
        # Get relevant clusters
        clusters = []
        if query.include_clusters:
            clusters = self._get_relevant_clusters(nodes, memory)
        
        # Build context
        context = ""
        if query.include_context:
            context = self._build_context(nodes, edges, clusters)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "context": context,
            "scores": scores,
        }
    
    def _expand_graph(
        self,
        initial_nodes: List[MemoryNode],
        memory: Memory,
        max_hops: int = 1,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Expand to related nodes via graph traversal."""
        node_ids = {n.id for n in initial_nodes}
        expanded_nodes = []
        related_edges = []
        
        for edge in memory.edges.values():
            if edge.source_id in node_ids or edge.target_id in node_ids:
                related_edges.append(edge)
                
                # Add connected nodes
                for connected_id in [edge.source_id, edge.target_id]:
                    if connected_id not in node_ids:
                        if connected_id in memory.nodes:
                            expanded_nodes.append(memory.nodes[connected_id])
                            node_ids.add(connected_id)
        
        return expanded_nodes, related_edges
    
    def _get_relevant_clusters(
        self,
        nodes: List[MemoryNode],
        memory: Memory,
    ) -> List[MemoryCluster]:
        """Get clusters containing the retrieved nodes."""
        relevant_clusters = []
        node_names = {n.name for n in nodes}
        
        for cluster in memory.clusters.values():
            # Check if cluster contains any of our nodes
            if any(name in node_names for name in cluster.entities):
                relevant_clusters.append(cluster)
        
        return relevant_clusters
    
    def _build_context(
        self,
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
        clusters: List[MemoryCluster],
    ) -> str:
        """Build context string from retrieved elements."""
        context_parts = []
        
        # Add entity descriptions
        if nodes:
            context_parts.append("Relevant Entities:")
            for node in nodes[:10]:  # Limit
                desc = node.description or node.name
                context_parts.append(f"- {node.name} ({node.entity_type}): {desc}")
        
        # Add relationships
        if edges:
            context_parts.append("\nRelationships:")
            for edge in edges[:10]:
                context_parts.append(
                    f"- {edge.source_id} --[{edge.relation_type}]--> {edge.target_id}"
                )
        
        # Add cluster summaries
        if clusters:
            context_parts.append("\nTopic Summaries:")
            for cluster in clusters[:3]:
                context_parts.append(f"- {cluster.summary}")
        
        return "\n".join(context_parts)


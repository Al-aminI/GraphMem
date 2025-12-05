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
        
        Uses HYBRID retrieval:
        1. EXACT NAME MATCH - Find entities mentioned by name in query
        2. SEMANTIC SEARCH - Find semantically similar entities
        3. GRAPH EXPANSION - Follow relationships from found entities
        
        Args:
            query: Query specification
            memory: Memory to search
        
        Returns:
            Dict with nodes, edges, clusters, and context
        """
        # Index memory for search
        self.semantic_search.index_nodes(list(memory.nodes.values()))
        
        # ===== STEP 1: EXACT NAME MATCH =====
        # Find entities that are EXPLICITLY mentioned in the query
        # This prevents retrieving noise entities!
        exact_matches = self._find_exact_name_matches(query.query, memory)
        
        # ===== STEP 2: SEMANTIC SEARCH =====
        node_results = self.semantic_search.search(
            query=query.query,
            top_k=query.top_k,
            min_similarity=query.min_similarity,
            filters=query.filters,
        )
        
        # Combine results: exact matches get HIGHEST priority
        nodes = []
        scores = {}
        
        # Add exact matches first with high score
        for node in exact_matches:
            if node.id not in scores:
                nodes.append(node)
                scores[node.id] = 1.0  # Perfect score for exact match
                logger.debug(f"Exact match: '{node.name}' in query")
        
        # Add semantic search results
        for node, score in node_results:
            if node.id not in scores:
                nodes.append(node)
                scores[node.id] = score
            else:
                # Boost score if also found by semantic search
                scores[node.id] = min(1.0, scores[node.id] + score * 0.2)
        
        # Expand via graph traversal - MORE HOPS for exact matches
        edges = []
        if nodes:
            # Use more hops if we have exact name matches (likely a specific query)
            max_hops = 3 if exact_matches else 1
            
            expanded_nodes, edges = self._expand_graph(
                initial_nodes=nodes,
                memory=memory,
                max_hops=max_hops,
            )
            
            # Add expanded nodes (with lower scores based on hop distance)
            for node in expanded_nodes:
                if node.id not in scores:
                    nodes.append(node)
                    scores[node.id] = 0.4  # Lower score for expanded
        
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
    
    def _find_exact_name_matches(
        self,
        query: str,
        memory: Memory,
    ) -> List[MemoryNode]:
        """
        Find entities that are EXPLICITLY mentioned by name in the query.
        
        This is CRITICAL for avoiding noise:
        - Query: "Who is the CEO of Helix Quantum Computing?"
        - Should prioritize "Helix Quantum Computing" entity over other companies
        
        Matches against:
        - Entity name
        - Entity aliases
        - Canonical name
        """
        import re
        
        query_lower = query.lower()
        matches = []
        matched_ids = set()
        
        for node in memory.nodes.values():
            # Check entity name
            if node.name.lower() in query_lower:
                if node.id not in matched_ids:
                    matches.append(node)
                    matched_ids.add(node.id)
                    continue
            
            # Check canonical name
            if node.canonical_name and node.canonical_name.lower() in query_lower:
                if node.id not in matched_ids:
                    matches.append(node)
                    matched_ids.add(node.id)
                    continue
            
            # Check aliases
            if hasattr(node, 'aliases') and node.aliases:
                for alias in node.aliases:
                    if len(alias) >= 3 and alias.lower() in query_lower:
                        if node.id not in matched_ids:
                            matches.append(node)
                            matched_ids.add(node.id)
                            break
        
        if matches:
            logger.info(f"🎯 Exact name matches: {[n.name for n in matches]}")
        
        return matches
    
    def _expand_graph(
        self,
        initial_nodes: List[MemoryNode],
        memory: Memory,
        max_hops: int = 1,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """
        Expand to related nodes via graph traversal.
        
        Supports multi-hop expansion for chain queries:
        A → B → C → D (3 hops)
        """
        all_node_ids = {n.id for n in initial_nodes}
        current_frontier = set(all_node_ids)
        expanded_nodes = []
        related_edges = []
        collected_edge_ids = set()
        
        for hop in range(max_hops):
            next_frontier = set()
            
            for edge in memory.edges.values():
                # Skip already collected edges
                if edge.id in collected_edge_ids:
                    continue
                    
                # Check if edge connects to current frontier
                if edge.source_id in current_frontier or edge.target_id in current_frontier:
                    related_edges.append(edge)
                    collected_edge_ids.add(edge.id)
                    
                    # Add connected nodes to next frontier
                    for connected_id in [edge.source_id, edge.target_id]:
                        if connected_id not in all_node_ids:
                            if connected_id in memory.nodes:
                                expanded_nodes.append(memory.nodes[connected_id])
                                all_node_ids.add(connected_id)
                                next_frontier.add(connected_id)
            
            # Move to next hop
            current_frontier = next_frontier
            if not current_frontier:
                break  # No more nodes to expand
        
        if expanded_nodes:
            logger.debug(f"Graph expansion: {len(expanded_nodes)} nodes in {max_hops} hops")
        
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


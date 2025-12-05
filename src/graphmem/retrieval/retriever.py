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
        llm=None,  # NEW: LLM for alias expansion
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
            llm: Optional LLM for alias expansion during retrieval
            top_k: Default number of results
            min_similarity: Minimum similarity threshold
            memory_id: Memory ID (used for Neo4j vector search)
            user_id: User ID for multi-tenant isolation
        """
        self.embeddings = embeddings
        self.store = store
        self.cache = cache
        self.llm = llm  # For LLM-based alias expansion
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
        
        # ===== STEP 1.5: LLM ALIAS EXPANSION (if no exact matches) =====
        # If we can't find exact matches, ask the LLM which entities match
        if not exact_matches and self.llm is not None:
            logger.debug("No exact matches found, using LLM alias expansion...")
            llm_matches = self._llm_expand_aliases(query.query, memory)
            if llm_matches:
                exact_matches = llm_matches
                logger.info(f"🤖 LLM found {len(llm_matches)} matching entities")
        
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
        
        # ===== NOISE FILTERING =====
        # If we have exact matches, be VERY selective about what else we include
        # This prevents noise entities from diluting the context
        if exact_matches:
            # Only add semantic results that are HIGHLY relevant (>0.75)
            for node, score in node_results:
                if node.id not in scores:
                    if score >= 0.75:  # High threshold when we have exact matches
                        nodes.append(node)
                        scores[node.id] = score * 0.8  # Discount vs exact matches
                else:
                    # Boost score if also found by semantic search
                    scores[node.id] = min(1.0, scores[node.id] + score * 0.2)
            logger.debug(f"Noise filtering: {len(exact_matches)} exact + {len(nodes)-len(exact_matches)} high-relevance")
        else:
            # No exact matches - use all semantic search results
            for node, score in node_results:
                if node.id not in scores:
                    nodes.append(node)
                    scores[node.id] = score
                else:
                    scores[node.id] = min(1.0, scores[node.id] + score * 0.2)
        
        # Expand via graph traversal - DEEP TRAVERSAL for chain queries
        edges = []
        if nodes:
            # Use 6+ hops for deep chain traversal (needed for multi-hop reasoning)
            max_hops = 6 if exact_matches else 2
            
            expanded_nodes, edges = self._expand_graph(
                initial_nodes=nodes,
                memory=memory,
                max_hops=max_hops,
            )
            
            # Add expanded nodes with decreasing scores by hop distance
            for node in expanded_nodes:
                if node.id not in scores:
                    nodes.append(node)
                    scores[node.id] = 0.5  # Expanded nodes still important
        
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
        
        COMPREHENSIVE ALIAS SEARCH:
        - Query: "What did Alexander Chen found?"
        - Entity has aliases: ["Dr. Chen", "The Quantum Pioneer", "A. Chen"]
        - Should find the entity even if query uses canonical name not in docs
        
        Also does REVERSE lookup:
        - If query contains ANY alias, find the canonical entity
        """
        import re
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        matches = []
        matched_ids = set()
        
        # Build reverse alias index: alias -> node
        alias_to_node = {}
        for node in memory.nodes.values():
            # Index by name
            alias_to_node[node.name.lower()] = node
            if node.canonical_name:
                alias_to_node[node.canonical_name.lower()] = node
            # Index by ALL aliases
            if hasattr(node, 'aliases') and node.aliases:
                for alias in node.aliases:
                    alias_to_node[alias.lower()] = node
        
        # PASS 1: Find entities whose name/alias appears in query
        for node in memory.nodes.values():
            if node.id in matched_ids:
                continue
                
            # Check entity name
            if node.name.lower() in query_lower:
                matches.append(node)
                matched_ids.add(node.id)
                continue
            
            # Check canonical name
            if node.canonical_name and node.canonical_name.lower() in query_lower:
                matches.append(node)
                matched_ids.add(node.id)
                continue
            
            # Check ALL aliases (comprehensive)
            if hasattr(node, 'aliases') and node.aliases:
                for alias in node.aliases:
                    alias_lower = alias.lower()
                    if len(alias) >= 3 and alias_lower in query_lower:
                        matches.append(node)
                        matched_ids.add(node.id)
                        break
        
        # PASS 2: Extract potential entity names from query and lookup
        # This catches cases where query uses a name that IS an alias
        potential_names = self._extract_potential_names(query)
        for name in potential_names:
            name_lower = name.lower()
            if name_lower in alias_to_node:
                node = alias_to_node[name_lower]
                if node.id not in matched_ids:
                    matches.append(node)
                    matched_ids.add(node.id)
                    logger.debug(f"Found via alias lookup: '{name}' → {node.name}")
        
        if matches:
            logger.info(f"🎯 Exact name matches: {[n.name for n in matches]}")
        
        return matches
    
    def _extract_potential_names(self, query: str) -> List[str]:
        """Extract potential entity names from a query."""
        import re
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        
        # Extract capitalized phrases (likely proper nouns)
        # e.g., "Alexander Chen", "Helix Quantum Computing"
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        
        # Extract title + name patterns
        # e.g., "Dr. Chen", "CEO Johnson"
        titled = re.findall(r'\b((?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.|CEO|CTO|CFO)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query)
        
        return list(set(quoted + capitalized + titled))
    
    def _llm_expand_aliases(self, query: str, memory: Memory) -> List[MemoryNode]:
        """
        Use LLM to find entities that match the query, even with different names.
        
        This is the ULTIMATE fallback - ask the LLM directly which entities
        from our knowledge base match what the user is asking about.
        """
        if self.llm is None:
            return []
        
        # Get all entity names from memory
        entity_list = []
        for node in list(memory.nodes.values())[:100]:  # Limit for prompt size
            aliases_str = f" (aliases: {', '.join(list(node.aliases)[:3])})" if node.aliases else ""
            entity_list.append(f"- {node.name}{aliases_str}: {node.entity_type}")
        
        if not entity_list:
            return []
        
        try:
            prompt = f"""Given this user query and list of entities in our knowledge base, identify which entities are RELEVANT to answering the query.

USER QUERY: {query}

ENTITIES IN KNOWLEDGE BASE:
{chr(10).join(entity_list[:50])}

INSTRUCTIONS:
- The query might use different names/aliases for the same entity
- "Alexander Chen" could be "Dr. Chen" or "A. Chen"
- "Helix Quantum Computing" could be "Helix QC" or "HQC"
- List ONLY the exact entity names from above that are relevant
- One entity name per line
- If no entities are relevant, respond with "NONE"

RELEVANT ENTITIES:"""

            response = self.llm.complete(prompt)
            
            # Parse response - extract entity names
            matches = []
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('- ').strip()
                if line and line.upper() != "NONE":
                    # Find matching node
                    for node in memory.nodes.values():
                        if node.name.lower() == line.lower():
                            matches.append(node)
                            logger.info(f"🤖 LLM alias match: query='{query[:30]}' → entity='{node.name}'")
                            break
            
            return matches
            
        except Exception as e:
            logger.warning(f"LLM alias expansion failed: {e}")
            return []
    
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
                
                # IMPORTANCE FILTERING: Skip decayed/superseded edges
                from graphmem.core.memory_types import MemoryImportance, MemoryState
                if edge.importance == MemoryImportance.EPHEMERAL:
                    continue  # Decayed edge - superseded by newer information
                if edge.state == MemoryState.ARCHIVED or edge.state == MemoryState.DELETED:
                    continue  # Archived/deleted - skip
                    
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


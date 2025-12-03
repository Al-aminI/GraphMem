"""
GraphMem Community Detector

Discovers communities (clusters) of related entities in the knowledge graph.
Communities enable efficient retrieval by grouping semantically related knowledge.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from graphmem.core.memory_types import (
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryImportance,
)

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Detects communities in knowledge graphs using graph algorithms.
    
    Generates summaries for each community using LLM to provide
    high-level understanding of knowledge domains.
    
    Algorithms:
    - greedy_modularity: Fast, good quality (default)
    - louvain: Higher quality, slower
    - label_propagation: Very fast, lower quality
    """
    
    def __init__(
        self,
        llm,
        max_cluster_size: int = 100,
        min_cluster_size: int = 2,
        algorithm: str = "greedy_modularity",
    ):
        """
        Initialize community detector.
        
        Args:
            llm: LLM provider for summary generation
            max_cluster_size: Maximum entities per cluster
            min_cluster_size: Minimum entities per cluster
            algorithm: Community detection algorithm
        """
        self.llm = llm
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
    
    def detect(
        self,
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
        memory_id: str,
    ) -> List[MemoryCluster]:
        """
        Detect communities in the graph.
        
        Args:
            nodes: List of nodes
            edges: List of edges
            memory_id: Parent memory ID
        
        Returns:
            List of detected communities with summaries
        """
        if not nodes:
            return []
        
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX not installed. Cannot detect communities.")
            return []
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_by_id = {n.id: n for n in nodes}
        for node in nodes:
            G.add_node(node.id, name=node.name, type=node.entity_type)
        
        # Add edges
        for edge in edges:
            if edge.source_id in node_by_id and edge.target_id in node_by_id:
                G.add_edge(
                    edge.source_id,
                    edge.target_id,
                    weight=edge.weight,
                    relation=edge.relation_type,
                )
        
        if not G.nodes():
            return []
        
        # Detect communities
        communities = self._detect_communities(G)
        
        if not communities:
            return []
        
        # Build clusters with summaries
        clusters = []
        entity_info: Dict[str, Set[int]] = defaultdict(set)
        
        for i, community in enumerate(communities):
            if len(community) < self.min_cluster_size:
                continue
            
            # Get entities in this community
            community_nodes = [node_by_id[nid] for nid in community if nid in node_by_id]
            
            if not community_nodes:
                continue
            
            # Track entity-community membership
            for node in community_nodes:
                entity_info[node.name].add(i)
            
            # Get edges within community
            community_edges = [
                e for e in edges
                if e.source_id in community and e.target_id in community
            ]
            
            # Generate summary
            summary = self._generate_summary(community_nodes, community_edges)
            
            # Calculate metrics
            coherence = self._calculate_coherence(G, community)
            density = self._calculate_density(G, community)
            
            # Determine importance from nodes (use .value for comparison)
            if community_nodes:
                importance = max(
                    community_nodes,
                    key=lambda n: n.importance.value,
                ).importance
            else:
                importance = MemoryImportance.MEDIUM
            
            cluster = MemoryCluster(
                id=i,
                summary=summary,
                entities=[n.name for n in community_nodes],
                edges=[e.id for e in community_edges],
                importance=importance,
                coherence_score=coherence,
                density=density,
                memory_id=memory_id,
                metadata={
                    "algorithm": self.algorithm,
                    "node_count": len(community_nodes),
                    "edge_count": len(community_edges),
                },
            )
            clusters.append(cluster)
        
        logger.info(f"Detected {len(clusters)} communities")
        return clusters
    
    def _detect_communities(self, G) -> List[Set]:
        """Run community detection algorithm."""
        import networkx as nx
        
        # Check for empty or too small graphs
        if len(G.nodes()) == 0:
            return []
        
        if len(G.nodes()) == 1:
            # Single node = single community
            return [set(G.nodes())]
        
        if len(G.edges()) == 0:
            # No edges - each node is its own community
            return [set([n]) for n in G.nodes()]
        
        try:
            if self.algorithm == "greedy_modularity":
                # greedy_modularity needs at least 2 nodes and 1 edge
                if len(G.nodes()) < 2 or len(G.edges()) < 1:
                    return [set(G.nodes())]
                communities = list(nx.community.greedy_modularity_communities(G))
            elif self.algorithm == "louvain":
                try:
                    communities_dict = nx.community.louvain_communities(G)
                    communities = list(communities_dict)
                except (AttributeError, ZeroDivisionError):
                    # Fallback if louvain not available or fails
                    if len(G.edges()) > 0:
                        communities = list(nx.community.greedy_modularity_communities(G))
                    else:
                        communities = [set(G.nodes())]
            elif self.algorithm == "label_propagation":
                communities = list(nx.community.label_propagation_communities(G))
            else:
                if len(G.edges()) > 0:
                    communities = list(nx.community.greedy_modularity_communities(G))
                else:
                    communities = [set(G.nodes())]
            
            # Split large communities
            result = []
            for community in communities:
                if len(community) > self.max_cluster_size:
                    # Split into smaller chunks
                    community_list = list(community)
                    for i in range(0, len(community_list), self.max_cluster_size):
                        result.append(set(community_list[i:i + self.max_cluster_size]))
                else:
                    result.append(community)
            
            return result
            
        except ZeroDivisionError:
            # NetworkX division by zero in modularity calculation
            logger.warning("Division by zero in community detection, returning all nodes as one community")
            return [set(G.nodes())]
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            # Return all nodes as single community as fallback
            if len(G.nodes()) > 0:
                return [set(G.nodes())]
            return []
    
    def _generate_summary(
        self,
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
    ) -> str:
        """Generate a summary for a community."""
        if not nodes:
            return "Empty community"
        
        # Build relationship description
        relationships = []
        for edge in edges[:20]:  # Limit for prompt length
            source_node = next((n for n in nodes if n.id == edge.source_id), None)
            target_node = next((n for n in nodes if n.id == edge.target_id), None)
            
            if source_node and target_node:
                desc = edge.description or f"{edge.relation_type}"
                relationships.append(
                    f"{source_node.name} → {edge.relation_type} → {target_node.name}: {desc}"
                )
        
        if not relationships:
            # No edges - just summarize entities
            entity_list = ", ".join(n.name for n in nodes[:10])
            return f"Group of related entities: {entity_list}"
        
        relationship_text = "\n".join(relationships)
        
        prompt = f"""You are analyzing relationships from a knowledge graph.
Create a coherent summary that captures all key facts, relationships, and context.

Relationships:
{relationship_text}

Write a comprehensive 2-3 sentence summary that:
1. Identifies the main entities and their roles
2. Describes how they are connected
3. Captures the key theme or topic of this group

Summary:"""
        
        try:
            response = self.llm.complete(prompt)
            summary = response.strip()
            
            # Clean up
            if summary.startswith("Summary:"):
                summary = summary[8:].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback to simple concatenation
            entity_names = [n.name for n in nodes[:5]]
            return f"Group containing: {', '.join(entity_names)}"
    
    def _calculate_coherence(self, G, community: Set) -> float:
        """Calculate how well-connected the community is internally."""
        if len(community) <= 1:
            return 1.0
        
        subgraph = G.subgraph(community)
        
        # Ratio of actual edges to possible edges
        actual_edges = subgraph.number_of_edges()
        max_edges = len(community) * (len(community) - 1) / 2
        
        if max_edges == 0:
            return 1.0
        
        return min(1.0, actual_edges / max_edges)
    
    def _calculate_density(self, G, community: Set) -> float:
        """Calculate edge density within the community."""
        if len(community) <= 1:
            return 1.0
        
        subgraph = G.subgraph(community)
        
        try:
            import networkx as nx
            return nx.density(subgraph)
        except:
            return 0.5


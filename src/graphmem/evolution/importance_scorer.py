"""
GraphMem Importance Scorer

Calculates and updates importance scores for memory elements.
Higher importance = slower decay, higher retrieval priority.

Implements the importance scoring formula from the paper:
ρ(e) = w1·f1(e) + w2·f2(e) + w3·f3(e) + w4·f4(e)

where:
- f1 = Temporal recency (exponential decay)
- f2 = Access frequency (logarithmic scaling)  
- f3 = PageRank centrality (structural importance)
- f4 = User feedback / explicit importance
"""

from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from graphmem.core.memory_types import (
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryImportance,
)

logger = logging.getLogger(__name__)


def _get_importance_value(importance) -> float:
    """Safely get numeric value from importance (enum or float)."""
    if hasattr(importance, 'value'):
        return importance.value
    return float(importance) if importance is not None else 5.0


class ImportanceScorer:
    """
    Scores memory elements based on multiple factors including PageRank.
    
    Scoring Factors (per paper):
    - f1: Temporal recency - Recent interactions indicate relevance
    - f2: Access frequency - Frequently accessed = more valuable  
    - f3: PageRank centrality - Well-connected entities are important
    - f4: User signals - Explicit importance markers
    
    The formula is:
    ρ(e) = w1·f1(e) + w2·f2(e) + w3·f3(e) + w4·f4(e)
    """
    
    # Cache for PageRank scores (recomputed periodically)
    _pagerank_cache: Dict[str, float] = {}
    _pagerank_cache_time: Optional[datetime] = None
    _pagerank_cache_ttl: float = 300.0  # 5 minutes
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3,
        pagerank_weight: float = 0.2,
        user_weight: float = 0.2,
        pagerank_damping: float = 0.85,
    ):
        """
        Initialize scorer with weight configuration matching paper.
        
        Args:
            recency_weight: w1 - Weight for recency score (default 0.3)
            frequency_weight: w2 - Weight for access frequency (default 0.3)
            pagerank_weight: w3 - Weight for PageRank centrality (default 0.2)
            user_weight: w4 - Weight for user-provided importance (default 0.2)
            pagerank_damping: Damping factor for PageRank (default 0.85)
        
        Note: Weights should sum to 1.0 as per paper Equation 7.
        """
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.pagerank_weight = pagerank_weight
        self.user_weight = user_weight
        self.pagerank_damping = pagerank_damping
        
        # For backwards compatibility, map old names
        self.connectivity_weight = pagerank_weight
        self.centrality_weight = 0.0  # Merged into PageRank
    
    def score_node(
        self,
        node: MemoryNode,
        all_edges: List[MemoryEdge],
        all_nodes: List[MemoryNode],
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate importance score for a node using the paper's formula.
        
        Formula: ρ(e) = w1·f1(e) + w2·f2(e) + w3·f3(e) + w4·f4(e)
        
        Args:
            node: Node to score
            all_edges: All edges in the memory
            all_nodes: All nodes in the memory
            current_time: Current time (defaults to now)
        
        Returns:
            Score from 0-10
        """
        current_time = current_time or datetime.utcnow()
        
        # f1: Temporal recency score (0-1) - Equation 8 in paper
        f1_recency = self._recency_score(node.accessed_at, current_time)
        
        # f2: Access frequency score (0-1) - Equation 9 in paper
        f2_frequency = self._frequency_score(node.access_count)
        
        # f3: PageRank centrality score (0-1) - Equation 10 in paper
        f3_pagerank = self._pagerank_score(node.id, all_nodes, all_edges)
        
        # f4: User/explicit importance (0-1)
        f4_user = _get_importance_value(node.importance) / 10.0
        
        # Weighted combination (Equation 7 in paper)
        score = (
            self.recency_weight * f1_recency +
            self.frequency_weight * f2_frequency +
            self.pagerank_weight * f3_pagerank +
            self.user_weight * f4_user
        )
        
        # Scale to 0-10
        return min(10.0, max(0.0, score * 10))
    
    def _pagerank_score(
        self,
        node_id: str,
        all_nodes: List[MemoryNode],
        all_edges: List[MemoryEdge],
    ) -> float:
        """
        Calculate PageRank centrality for a node.
        
        Uses NetworkX's PageRank implementation with caching for efficiency.
        Implements f3(e) = PageRank(e, G) from the paper.
        
        Args:
            node_id: ID of node to score
            all_nodes: All nodes in memory
            all_edges: All edges in memory
        
        Returns:
            PageRank score normalized to [0, 1]
        """
        # Check cache validity
        now = datetime.utcnow()
        if (
            self._pagerank_cache
            and self._pagerank_cache_time
            and (now - self._pagerank_cache_time).total_seconds() < self._pagerank_cache_ttl
            and node_id in self._pagerank_cache
        ):
            return self._pagerank_cache.get(node_id, 0.0)
        
        # Need to recompute PageRank
        try:
            import networkx as nx
            
            # Build NetworkX graph
            G = nx.DiGraph()
            
            # Add all nodes
            for node in all_nodes:
                G.add_node(node.id)
            
            # Add all edges
            for edge in all_edges:
                if edge.source_id in G.nodes and edge.target_id in G.nodes:
                    G.add_edge(
                        edge.source_id,
                        edge.target_id,
                        weight=edge.weight * edge.confidence,
                    )
            
            # Handle empty or disconnected graph
            if len(G.nodes) == 0:
                return 0.0
            
            # Compute PageRank with damping factor from paper (0.85)
            try:
                pagerank_scores = nx.pagerank(
                    G,
                    alpha=self.pagerank_damping,
                    weight='weight',
                    max_iter=100,
                )
            except nx.PowerIterationFailedConvergence:
                # Fallback to simpler computation
                pagerank_scores = {n: 1.0 / len(G.nodes) for n in G.nodes}
            
            # Normalize to [0, 1] range
            max_pr = max(pagerank_scores.values()) if pagerank_scores else 1.0
            if max_pr > 0:
                normalized_scores = {
                    k: v / max_pr for k, v in pagerank_scores.items()
                }
            else:
                normalized_scores = pagerank_scores
            
            # Update cache
            self._pagerank_cache = normalized_scores
            self._pagerank_cache_time = now
            
            return normalized_scores.get(node_id, 0.0)
            
        except ImportError:
            logger.warning("NetworkX not installed, falling back to degree centrality")
            return self._connectivity_score(node_id, all_edges)
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}, using fallback")
            return self._connectivity_score(node_id, all_edges)
    
    def score_edge(
        self,
        edge: MemoryEdge,
        source_node: Optional[MemoryNode],
        target_node: Optional[MemoryNode],
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate importance score for an edge.
        
        Args:
            edge: Edge to score
            source_node: Source node
            target_node: Target node
            current_time: Current time
        
        Returns:
            Score from 0-10
        """
        current_time = current_time or datetime.utcnow()
        
        # Base score from edge properties
        recency = self._recency_score(edge.accessed_at, current_time)
        frequency = self._frequency_score(edge.access_count)
        
        # Edge strength factors
        weight_factor = min(1.0, edge.weight / 5.0)
        confidence_factor = edge.confidence
        
        # Node importance affects edge importance
        node_factor = 0.5
        if source_node and target_node:
            node_factor = (
                _get_importance_value(source_node.importance) / 10.0 +
                _get_importance_value(target_node.importance) / 10.0
            ) / 2
        
        # Combine factors
        score = (
            0.25 * recency +
            0.2 * frequency +
            0.2 * weight_factor +
            0.15 * confidence_factor +
            0.2 * node_factor
        )
        
        return min(10.0, max(0.0, score * 10))
    
    def update_importance(
        self,
        node: MemoryNode,
        all_edges: List[MemoryEdge],
        all_nodes: List[MemoryNode],
    ) -> MemoryImportance:
        """
        Update and return new importance level for a node.
        """
        score = self.score_node(node, all_edges, all_nodes)
        return MemoryImportance.from_score(score)
    
    def _recency_score(
        self,
        accessed_at: datetime,
        current_time: datetime,
        half_life_days: float = 30.0,
    ) -> float:
        """
        Calculate recency score using exponential decay.
        
        Recent memories score higher.
        """
        age = current_time - accessed_at
        age_days = age.total_seconds() / 86400
        
        # Exponential decay
        decay = math.exp(-0.693 * age_days / half_life_days)
        return decay
    
    def _frequency_score(
        self,
        access_count: int,
        saturation_point: int = 100,
    ) -> float:
        """
        Calculate frequency score with diminishing returns.
        
        More accesses = higher score, but with saturation.
        """
        if access_count <= 0:
            return 0.0
        
        # Logarithmic scaling with saturation
        return min(1.0, math.log(1 + access_count) / math.log(1 + saturation_point))
    
    def _connectivity_score(
        self,
        node_id: str,
        edges: List[MemoryEdge],
        max_connections: int = 50,
    ) -> float:
        """
        Calculate connectivity score based on edge count.
        
        Well-connected nodes are more important.
        """
        connection_count = sum(
            1 for e in edges
            if e.source_id == node_id or e.target_id == node_id
        )
        
        # Logarithmic scaling
        if connection_count <= 0:
            return 0.0
        
        return min(1.0, math.log(1 + connection_count) / math.log(1 + max_connections))
    
    def _centrality_score(
        self,
        node: MemoryNode,
        all_nodes: List[MemoryNode],
        all_edges: List[MemoryEdge],
    ) -> float:
        """
        Calculate semantic centrality.
        
        Nodes that connect different clusters are more important.
        """
        # Count unique connected nodes
        connected = set()
        for edge in all_edges:
            if edge.source_id == node.id:
                connected.add(edge.target_id)
            elif edge.target_id == node.id:
                connected.add(edge.source_id)
        
        if not connected:
            return 0.0
        
        # Check how many different entity types are connected
        connected_types = set()
        for other_id in connected:
            for other_node in all_nodes:
                if other_node.id == other_id:
                    connected_types.add(other_node.entity_type)
                    break
        
        # More diverse connections = higher centrality
        type_diversity = len(connected_types) / max(5, len(set(n.entity_type for n in all_nodes)))
        connection_ratio = len(connected) / max(10, len(all_nodes))
        
        return min(1.0, (type_diversity + connection_ratio) / 2)


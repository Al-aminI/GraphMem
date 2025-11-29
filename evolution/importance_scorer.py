"""
GraphMem Importance Scorer

Calculates and updates importance scores for memory elements.
Higher importance = slower decay, higher retrieval priority.
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


class ImportanceScorer:
    """
    Scores memory elements based on various factors.
    
    Scoring Factors:
    - Recency: How recently the memory was accessed
    - Frequency: How often the memory is accessed
    - Connectivity: How well-connected in the graph
    - Semantic centrality: How central to core topics
    - User signals: Explicit importance markers
    """
    
    def __init__(
        self,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.25,
        connectivity_weight: float = 0.2,
        centrality_weight: float = 0.15,
        user_weight: float = 0.1,
    ):
        """
        Initialize scorer with weight configuration.
        
        Args:
            recency_weight: Weight for recency score
            frequency_weight: Weight for access frequency
            connectivity_weight: Weight for graph connectivity
            centrality_weight: Weight for semantic centrality
            user_weight: Weight for user-provided importance
        """
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.connectivity_weight = connectivity_weight
        self.centrality_weight = centrality_weight
        self.user_weight = user_weight
    
    def score_node(
        self,
        node: MemoryNode,
        all_edges: List[MemoryEdge],
        all_nodes: List[MemoryNode],
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate importance score for a node.
        
        Args:
            node: Node to score
            all_edges: All edges in the memory
            all_nodes: All nodes in the memory
            current_time: Current time (defaults to now)
        
        Returns:
            Score from 0-10
        """
        current_time = current_time or datetime.utcnow()
        
        # Recency score (0-1)
        recency = self._recency_score(node.accessed_at, current_time)
        
        # Frequency score (0-1)
        frequency = self._frequency_score(node.access_count)
        
        # Connectivity score (0-1)
        connectivity = self._connectivity_score(node.id, all_edges)
        
        # Centrality score (0-1)
        centrality = self._centrality_score(node, all_nodes, all_edges)
        
        # User importance (0-1)
        user = node.importance.value / 10.0
        
        # Weighted combination
        score = (
            self.recency_weight * recency +
            self.frequency_weight * frequency +
            self.connectivity_weight * connectivity +
            self.centrality_weight * centrality +
            self.user_weight * user
        )
        
        # Scale to 0-10
        return min(10.0, max(0.0, score * 10))
    
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
                source_node.importance.value / 10.0 +
                target_node.importance.value / 10.0
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


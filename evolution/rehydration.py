"""
GraphMem Rehydration

Restores and strengthens memories based on context.
Like recalling a forgotten memory when given the right cue.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryState,
    EvolutionEvent,
    EvolutionType,
)

logger = logging.getLogger(__name__)


class GraphRehydration:
    """
    Rehydrates (restores) memories based on context.
    
    When a query or context relates to archived/weakened memories,
    rehydration can:
    1. Restore archived memories to active state
    2. Strengthen weakened but relevant memories
    3. Re-integrate memories into the active graph
    """
    
    def __init__(
        self,
        embeddings,
        rehydration_threshold: float = 0.75,
        max_rehydrations: int = 100,
        strength_boost: float = 1.5,
    ):
        """
        Initialize rehydration handler.
        
        Args:
            embeddings: Embedding provider for similarity matching
            rehydration_threshold: Minimum similarity for rehydration
            max_rehydrations: Maximum memories to rehydrate per call
            strength_boost: Factor to boost rehydrated memory strength
        """
        self.embeddings = embeddings
        self.rehydration_threshold = rehydration_threshold
        self.max_rehydrations = max_rehydrations
        self.strength_boost = strength_boost
    
    def rehydrate(
        self,
        memory: Memory,
        context: str,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Rehydrate memories based on context.
        
        Args:
            memory: Memory to rehydrate
            context: Context/query to match against
            max_nodes: Maximum nodes to consider
        
        Returns:
            Statistics about rehydration
        """
        if not context or not context.strip():
            return {"rehydrated": 0, "restored": 0, "strengthened": 0}
        
        # Get context embedding
        try:
            context_embedding = self.embeddings.embed_text(context)
            if not context_embedding:
                return {"rehydrated": 0, "restored": 0, "strengthened": 0}
            context_vector = np.array(context_embedding)
        except Exception as e:
            logger.error(f"Failed to get context embedding: {e}")
            return {"rehydrated": 0, "restored": 0, "strengthened": 0}
        
        # Find relevant memories
        rehydrated_count = 0
        restored_count = 0
        strengthened_count = 0
        
        # Score all nodes
        scored_nodes = []
        for node in memory.nodes.values():
            score = self._score_relevance(node, context_vector)
            if score >= self.rehydration_threshold:
                scored_nodes.append((node, score))
        
        # Sort by score and limit
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        scored_nodes = scored_nodes[:max_nodes]
        
        # Rehydrate relevant nodes
        for node, score in scored_nodes:
            if node.state == MemoryState.ARCHIVED:
                # Restore archived memory
                memory.nodes[node.id] = node.evolve(
                    state=MemoryState.ACTIVE,
                    accessed_at=datetime.utcnow(),
                    access_count=node.access_count + 1,
                )
                restored_count += 1
                rehydrated_count += 1
                
            elif node.state == MemoryState.DECAYING:
                # Restore decaying memory
                memory.nodes[node.id] = node.evolve(
                    state=MemoryState.ACTIVE,
                    accessed_at=datetime.utcnow(),
                    access_count=node.access_count + 1,
                )
                restored_count += 1
                rehydrated_count += 1
                
            elif node.state == MemoryState.ACTIVE:
                # Strengthen active memory
                memory.nodes[node.id] = node.evolve(
                    accessed_at=datetime.utcnow(),
                    access_count=node.access_count + 1,
                )
                strengthened_count += 1
                rehydrated_count += 1
            
            if rehydrated_count >= self.max_rehydrations:
                break
        
        # Also rehydrate connected edges
        edge_rehydrated = self._rehydrate_edges(memory, scored_nodes)
        
        logger.info(
            f"Rehydration complete: {rehydrated_count} nodes "
            f"({restored_count} restored, {strengthened_count} strengthened), "
            f"{edge_rehydrated} edges"
        )
        
        return {
            "rehydrated": rehydrated_count,
            "restored": restored_count,
            "strengthened": strengthened_count,
            "edges_rehydrated": edge_rehydrated,
        }
    
    def _score_relevance(
        self,
        node: MemoryNode,
        context_vector: np.ndarray,
    ) -> float:
        """Score how relevant a node is to the context."""
        # Get node text
        text = node.description or node.name
        
        try:
            node_embedding = self.embeddings.embed_text(text)
            if not node_embedding:
                return 0.0
            
            node_vector = np.array(node_embedding)
            
            # Cosine similarity
            similarity = self._cosine_similarity(context_vector, node_vector)
            
            # Boost for archived memories (they need more help)
            if node.state == MemoryState.ARCHIVED:
                similarity *= 1.1
            
            return similarity
            
        except Exception as e:
            logger.debug(f"Failed to score node {node.id}: {e}")
            return 0.0
    
    def _rehydrate_edges(
        self,
        memory: Memory,
        scored_nodes: List[tuple],
    ) -> int:
        """Rehydrate edges connected to rehydrated nodes."""
        rehydrated_node_ids = {node.id for node, _ in scored_nodes}
        rehydrated_count = 0
        
        for edge_id, edge in memory.edges.items():
            if edge.state in (MemoryState.ACTIVE,):
                continue
            
            # Check if connected to rehydrated nodes
            if edge.source_id in rehydrated_node_ids or edge.target_id in rehydrated_node_ids:
                memory.edges[edge_id] = edge.evolve(
                    state=MemoryState.ACTIVE,
                    accessed_at=datetime.utcnow(),
                    access_count=edge.access_count + 1,
                )
                rehydrated_count += 1
        
        return rehydrated_count
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


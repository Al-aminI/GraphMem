"""
GraphMem Memory Decay

Implements forgetting mechanisms for memory management.
Like human memory, less important and less accessed memories decay over time.
"""

from __future__ import annotations
import logging
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryImportance,
    MemoryState,
    EvolutionEvent,
    EvolutionType,
)
from graphmem.evolution.importance_scorer import ImportanceScorer

logger = logging.getLogger(__name__)


class MemoryDecay:
    """
    Handles memory decay (forgetting) over time.
    
    Decay Behavior:
    - Critical memories never decay
    - Higher importance = slower decay
    - Frequently accessed memories decay slower
    - Memories can be archived (not deleted) for potential rehydration
    """
    
    def __init__(
        self,
        half_life_days: float = 30.0,
        min_importance_to_keep: MemoryImportance = MemoryImportance.VERY_LOW,
        archive_threshold: float = 0.2,  # Strength at which to archive
        delete_threshold: float = 0.05,  # Strength at which to delete
    ):
        """
        Initialize decay handler.
        
        Args:
            half_life_days: Time for memory strength to halve
            min_importance_to_keep: Minimum importance level to prevent decay
            archive_threshold: Strength threshold for archiving
            delete_threshold: Strength threshold for deletion
        """
        self.half_life_days = half_life_days
        self.min_importance_to_keep = min_importance_to_keep
        self.archive_threshold = archive_threshold
        self.delete_threshold = delete_threshold
        self.importance_scorer = ImportanceScorer()
    
    def apply_decay(
        self,
        memory: Memory,
        current_time: Optional[datetime] = None,
    ) -> List[EvolutionEvent]:
        """
        Apply decay to all memory elements.
        
        Args:
            memory: Memory to decay
            current_time: Current time (defaults to now)
        
        Returns:
            List of evolution events describing what was decayed
        """
        current_time = current_time or datetime.utcnow()
        events = []
        
        # Decay nodes
        node_events = self._decay_nodes(memory, current_time)
        events.extend(node_events)
        
        # Decay edges
        edge_events = self._decay_edges(memory, current_time)
        events.extend(edge_events)
        
        logger.info(f"Applied decay: {len(events)} elements affected")
        return events
    
    def _decay_nodes(
        self,
        memory: Memory,
        current_time: datetime,
    ) -> List[EvolutionEvent]:
        """Decay nodes based on age and importance."""
        events = []
        nodes_to_archive = []
        nodes_to_delete = []
        
        all_edges = list(memory.edges.values())
        all_nodes = list(memory.nodes.values())
        
        for node_id, node in list(memory.nodes.items()):
            # Skip if already archived or deleted
            if node.state in (MemoryState.ARCHIVED, MemoryState.DELETED):
                continue
            
            # Critical memories never decay
            if node.importance == MemoryImportance.CRITICAL:
                continue
            
            # Calculate decay factor
            strength = self._calculate_strength(node, current_time)
            
            # Determine action
            if strength <= self.delete_threshold:
                nodes_to_delete.append(node_id)
            elif strength <= self.archive_threshold:
                nodes_to_archive.append(node_id)
            else:
                # Update importance based on decay
                new_importance = self.importance_scorer.update_importance(
                    node, all_edges, all_nodes
                )
                
                if new_importance != node.importance:
                    memory.nodes[node_id] = node.evolve(importance=new_importance)
        
        # Archive nodes
        for node_id in nodes_to_archive:
            if memory.nodes[node_id].importance.value >= self.min_importance_to_keep.value:
                continue  # Don't archive if importance is high enough
            
            before_state = memory.nodes[node_id].to_dict()
            memory.nodes[node_id] = memory.nodes[node_id].evolve(
                state=MemoryState.ARCHIVED
            )
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.DECAY,
                memory_id=memory.id,
                affected_nodes=[node_id],
                before_state={"state": before_state.get("state")},
                after_state={"state": "ARCHIVED"},
                reason="Memory strength below archive threshold",
            ))
        
        # Delete nodes (soft delete)
        for node_id in nodes_to_delete:
            if memory.nodes[node_id].importance.value >= self.min_importance_to_keep.value:
                continue
            
            before_state = memory.nodes[node_id].to_dict()
            memory.nodes[node_id] = memory.nodes[node_id].evolve(
                state=MemoryState.DELETED
            )
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.PRUNING,
                memory_id=memory.id,
                affected_nodes=[node_id],
                before_state={"state": before_state.get("state")},
                after_state={"state": "DELETED"},
                reason="Memory strength below delete threshold",
            ))
        
        return events
    
    def _decay_edges(
        self,
        memory: Memory,
        current_time: datetime,
    ) -> List[EvolutionEvent]:
        """Decay edges based on age and strength."""
        events = []
        edges_to_weaken = []
        edges_to_delete = []
        
        for edge_id, edge in list(memory.edges.items()):
            if edge.state in (MemoryState.ARCHIVED, MemoryState.DELETED):
                continue
            
            if edge.importance == MemoryImportance.CRITICAL:
                continue
            
            # Check if source or target is deleted
            source_deleted = (
                edge.source_id in memory.nodes and
                memory.nodes[edge.source_id].state == MemoryState.DELETED
            )
            target_deleted = (
                edge.target_id in memory.nodes and
                memory.nodes[edge.target_id].state == MemoryState.DELETED
            )
            
            if source_deleted or target_deleted:
                edges_to_delete.append(edge_id)
                continue
            
            # Calculate decay
            strength = self._calculate_edge_strength(edge, current_time)
            
            if strength <= self.delete_threshold:
                edges_to_delete.append(edge_id)
            elif strength < 0.5:
                edges_to_weaken.append((edge_id, strength))
        
        # Weaken edges
        for edge_id, new_strength in edges_to_weaken:
            old_weight = memory.edges[edge_id].weight
            new_weight = old_weight * new_strength
            
            memory.edges[edge_id] = memory.edges[edge_id].evolve(weight=new_weight)
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.DECAY,
                memory_id=memory.id,
                affected_edges=[edge_id],
                before_state={"weight": old_weight},
                after_state={"weight": new_weight},
                reason="Edge decay over time",
            ))
        
        # Delete edges
        for edge_id in edges_to_delete:
            before_state = memory.edges[edge_id].to_dict()
            memory.edges[edge_id] = memory.edges[edge_id].evolve(
                state=MemoryState.DELETED
            )
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.PRUNING,
                memory_id=memory.id,
                affected_edges=[edge_id],
                before_state={"state": before_state.get("state")},
                after_state={"state": "DELETED"},
                reason="Edge strength below threshold or connected to deleted node",
            ))
        
        return events
    
    def _calculate_strength(
        self,
        node: MemoryNode,
        current_time: datetime,
    ) -> float:
        """
        Calculate current strength of a memory node.
        
        Returns value 0-1 where 1 is full strength.
        """
        # Time since last access
        age = current_time - node.accessed_at
        age_days = age.total_seconds() / 86400
        
        # Importance modifier (higher importance = slower decay)
        importance_factor = 0.5 + (node.importance.value / 20.0)  # 0.5 to 1.0
        
        # Access count modifier (more access = slower decay)
        access_factor = min(1.0, 0.5 + math.log(1 + node.access_count) / 10)
        
        # Effective half-life
        effective_half_life = self.half_life_days * importance_factor * access_factor
        
        # Exponential decay
        strength = math.exp(-0.693 * age_days / effective_half_life)
        
        return strength
    
    def _calculate_edge_strength(
        self,
        edge: MemoryEdge,
        current_time: datetime,
    ) -> float:
        """Calculate current strength of an edge."""
        age = current_time - edge.accessed_at
        age_days = age.total_seconds() / 86400
        
        # Edge weight and confidence affect decay
        weight_factor = min(1.0, edge.weight / 5.0)
        confidence_factor = edge.confidence
        
        # Effective half-life
        effective_half_life = self.half_life_days * weight_factor * confidence_factor
        
        # Exponential decay
        strength = math.exp(-0.693 * age_days / max(1.0, effective_half_life))
        
        return strength


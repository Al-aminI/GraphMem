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


def _get_importance_value(importance) -> float:
    """Safely get numeric value from importance (enum or float)."""
    if hasattr(importance, 'value'):
        return importance.value
    return float(importance) if importance is not None else 5.0


class MemoryDecay:
    """
    Handles memory decay (forgetting) over time.
    
    Decay Behavior:
    - Critical memories never decay
    - Higher importance = slower decay
    - Frequently accessed memories decay slower
    - Memories can be archived (not deleted) for potential rehydration
    - LLM-BASED REASONING for intelligent decay decisions
    """
    
    def __init__(
        self,
        llm=None,  # NEW: LLM for intelligent decay decisions
        half_life_days: float = 30.0,
        min_importance_to_keep: MemoryImportance = MemoryImportance.VERY_LOW,
        archive_threshold: float = 0.2,  # Strength at which to archive
        delete_threshold: float = 0.05,  # Strength at which to delete
    ):
        """
        Initialize decay handler.
        
        Args:
            llm: Optional LLM for intelligent decay reasoning
            half_life_days: Time for memory strength to halve
            min_importance_to_keep: Minimum importance level to prevent decay
            archive_threshold: Strength threshold for archiving
            delete_threshold: Strength threshold for deletion
        """
        self.llm = llm
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
        
        # LLM-based decay reasoning for temporal conflicts
        if self.llm:
            llm_events = self._llm_based_decay(memory, current_time)
            events.extend(llm_events)
        
        logger.info(f"Applied decay: {len(events)} elements affected")
        return events
    
    def _llm_based_decay(
        self,
        memory: Memory,
        current_time: datetime,
    ) -> List[EvolutionEvent]:
        """
        LLM-BASED DECAY REASONING
        
        Ask the LLM to identify:
        1. Outdated facts that conflict with newer ones
        2. Temporal relationships that have ended
        3. Redundant information that can be forgotten
        4. CONFLICTING relationships (same source, same relation type, different targets)
        """
        events = []
        
        # AGGRESSIVE: Analyze ALL relationships, not just ones with temporal validity
        # Look for CONFLICTS: same entity with conflicting information
        all_edges = []
        for edge in memory.edges.values():
            source_node = memory.nodes.get(edge.source_id)
            target_node = memory.nodes.get(edge.target_id)
            source_name = source_node.name if source_node else edge.source_id
            target_name = target_node.name if target_node else edge.target_id
            
            valid_from = "unknown"
            valid_until = "present"
            if edge.valid_from:
                valid_from = edge.valid_from.strftime("%Y") if hasattr(edge.valid_from, 'strftime') else str(edge.valid_from)
            if edge.valid_until:
                valid_until = edge.valid_until.strftime("%Y") if hasattr(edge.valid_until, 'strftime') else str(edge.valid_until)
            
            all_edges.append({
                "id": edge.id,
                "source": source_name,
                "target": target_name,
                "relation": edge.relation_type,
                "description": f"{source_name} --[{edge.relation_type}]--> {target_name}",
                "valid_from": valid_from,
                "valid_until": valid_until,
                "edge": edge,
            })
        
        if len(all_edges) < 2:
            return events  # Need at least 2 edges to find conflicts
        
        # Group edges by source + relation type to find conflicts
        from collections import defaultdict
        edge_groups = defaultdict(list)
        for edge_info in all_edges:
            key = f"{edge_info['source'].lower()}|{edge_info['relation'].lower()}"
            edge_groups[key].append(edge_info)
        
        # Find groups with potential conflicts (same source+relation, different targets)
        # PRIORITY-BASED CONFLICT RESOLUTION: Lower priority edges should decay
        conflict_candidates = []
        priority_based_decays = []
        
        for key, edges in edge_groups.items():
            if len(edges) > 1:
                # Multiple targets for same source+relation = potential conflict
                targets = set(e['target'].lower() for e in edges)
                if len(targets) > 1:
                    # PRIORITY RESOLUTION: Sort by priority, decay all but highest
                    sorted_edges = sorted(
                        edges, 
                        key=lambda e: e['edge'].properties.get('priority', 0),
                        reverse=True
                    )
                    
                    # Keep highest priority, mark others for decay
                    if len(sorted_edges) > 1:
                        highest = sorted_edges[0]
                        highest_priority = highest['edge'].properties.get('priority', 0)
                        
                        for lower_edge in sorted_edges[1:]:
                            lower_priority = lower_edge['edge'].properties.get('priority', 0)
                            if lower_priority < highest_priority:
                                logger.info(
                                    f"🔄 Priority decay: {lower_edge['description']} "
                                    f"(priority={lower_priority}) superseded by "
                                    f"{highest['description']} (priority={highest_priority})"
                                )
                                priority_based_decays.append(lower_edge)
                            else:
                                # Same priority - send to LLM for resolution
                                conflict_candidates.append(lower_edge)
                        
                        conflict_candidates.append(highest)  # Include for context
        
        # Add priority-based decays as events - LOWER IMPORTANCE instead of deleting
        from graphmem.core.memory_types import MemoryImportance, MemoryState
        
        for decay_edge in priority_based_decays:
            edge = decay_edge['edge']
            old_importance = edge.importance
            
            # Lower importance to EPHEMERAL (will be filtered in retrieval)
            if edge.id in memory.edges:
                memory.edges[edge.id] = edge.evolve(
                    importance=MemoryImportance.EPHEMERAL,
                    state=MemoryState.ARCHIVED,  # Mark as archived, not deleted
                )
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.DECAY,
                memory_id=memory.id,
                affected_edges=[edge.id],
                before_state={"importance": old_importance.name},
                after_state={"importance": "EPHEMERAL", "state": "ARCHIVED"},
                reason=f"Superseded by higher-priority fact (conflict resolution)",
            ))
        
        # If no conflict candidates, fall back to temporal analysis
        temporal_edges = [e for e in all_edges if e['valid_from'] != 'unknown' or e['valid_until'] != 'present']
        
        edges_to_analyze = conflict_candidates if conflict_candidates else temporal_edges[:30]
        
        if len(edges_to_analyze) < 2:
            return events
        
        logger.info(f"🧠 LLM decay analyzing {len(edges_to_analyze)} relationships for conflicts...")
        
        # Build edge descriptions for LLM
        edge_descriptions = []
        for i, e in enumerate(edges_to_analyze[:40]):  # Limit for prompt size
            temporal_info = f"[valid: {e['valid_from']} to {e['valid_until']}]"
            edge_descriptions.append(f"{i+1}. {e['description']} {temporal_info}")
        
        # Identify if we're looking at conflicts
        has_conflicts = len(conflict_candidates) > 0
        
        prompt = f"""Analyze these knowledge graph relationships and identify which ones are OUTDATED, SUPERSEDED, or should DECAY.

RELATIONSHIPS TO ANALYZE:
{chr(10).join(edge_descriptions)}

CURRENT DATE: {current_time.strftime("%Y-%m-%d")}

{"NOTE: Some relationships have CONFLICTING targets for the same source and relation type. The NEWER information should supersede the OLDER." if has_conflicts else ""}

Identify relationships that:
1. Are OUTDATED (superseded by newer information)
2. CONFLICT with more recent relationships (e.g., old CEO vs new CEO, old location vs new)
3. Represent historical facts that have been REPLACED

CRITICAL: When the SAME entity has MULTIPLE values for the SAME relation (e.g., "Person X is CEO of Company A" AND "Person Y is CEO of Company A"), the OLDER one should decay.

For each outdated relationship, output:
<number>|DECAY|<reason>

Only list relationships that should decay. If unsure, don't decay.

OUTPUT:"""

        try:
            response = self.llm.complete(prompt)
            decay_count = 0
            
            for line in response.strip().split('\n'):
                if '|DECAY|' not in line:
                    continue
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        edge_num = int(parts[0].strip()) - 1
                        reason = parts[2].strip()
                        
                        if 0 <= edge_num < len(edges_to_analyze):
                            edge_info = edges_to_analyze[edge_num]
                            edge = edge_info["edge"]
                            
                            # Mark the edge as decayed
                            logger.info(f"🧠 LLM decay: {edge_info['description']} - {reason}")
                            
                            events.append(EvolutionEvent(
                                evolution_type=EvolutionType.DECAY,
                                memory_id=memory.id,
                                affected_edges=[edge.id],
                                before_state={"state": "active"},
                                after_state={"state": "decayed"},
                                reason=f"LLM conflict resolution: {reason}",
                            ))
                            decay_count += 1
                    except (ValueError, IndexError):
                        continue
            
            if decay_count > 0:
                logger.info(f"🧠 LLM identified {decay_count} outdated relationships")
                        
        except Exception as e:
            logger.warning(f"LLM decay reasoning failed: {e}")
        
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
            if _get_importance_value(memory.nodes[node_id].importance) >= _get_importance_value(self.min_importance_to_keep):
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
            if _get_importance_value(memory.nodes[node_id].importance) >= _get_importance_value(self.min_importance_to_keep):
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
        importance_factor = 0.5 + (_get_importance_value(node.importance) / 20.0)  # 0.5 to 1.0
        
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


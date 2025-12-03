"""
GraphMem Memory Consolidation

Merges related memories to create stronger, more coherent knowledge.
Like human memory consolidation during sleep.
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryImportance,
    EvolutionEvent,
    EvolutionType,
)

logger = logging.getLogger(__name__)


class MemoryConsolidation:
    """
    Consolidates similar memories into stronger, unified representations.
    
    Consolidation Types:
    1. Entity Merging: Merge duplicate/similar entities
    2. Edge Strengthening: Strengthen frequently co-occurring relationships
    3. Cluster Refinement: Improve community summaries
    4. Knowledge Synthesis: Create new memories from patterns
    """
    
    def __init__(
        self,
        embeddings,
        similarity_threshold: float = 0.85,
        min_occurrences_to_merge: int = 2,
        synthesis_enabled: bool = True,
    ):
        """
        Initialize consolidation handler.
        
        Args:
            embeddings: Embedding provider
            similarity_threshold: Threshold for considering entities similar
            min_occurrences_to_merge: Minimum occurrences before merging
            synthesis_enabled: Whether to create synthesized memories
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.min_occurrences_to_merge = min_occurrences_to_merge
        self.synthesis_enabled = synthesis_enabled
    
    def consolidate(
        self,
        memory: Memory,
    ) -> List[EvolutionEvent]:
        """
        Consolidate memories.
        
        Args:
            memory: Memory to consolidate
        
        Returns:
            List of evolution events
        """
        events = []
        
        # 1. Find and merge similar entities
        merge_events = self._consolidate_entities(memory)
        events.extend(merge_events)
        
        # 2. Strengthen frequently co-occurring edges
        reinforce_events = self._reinforce_edges(memory)
        events.extend(reinforce_events)
        
        # 3. Synthesize new knowledge (optional)
        if self.synthesis_enabled:
            synthesis_events = self._synthesize_knowledge(memory)
            events.extend(synthesis_events)
        
        logger.info(f"Consolidation complete: {len(events)} events")
        return events
    
    def _consolidate_entities(
        self,
        memory: Memory,
    ) -> List[EvolutionEvent]:
        """Find and merge similar entities."""
        events = []
        
        nodes = list(memory.nodes.values())
        if len(nodes) < 2:
            return events
        
        # Group nodes by entity type for efficiency
        type_groups: Dict[str, List[MemoryNode]] = {}
        for node in nodes:
            key = node.entity_type.lower() if node.entity_type else "unknown"
            type_groups.setdefault(key, []).append(node)
        
        # Find merge candidates within each type
        merge_groups: List[Set[str]] = []
        processed: Set[str] = set()
        
        for entity_type, type_nodes in type_groups.items():
            if len(type_nodes) < 2:
                continue
            
            # Get embeddings for all nodes in this type
            embeddings_map = {}
            for node in type_nodes:
                text = node.description or node.name
                try:
                    emb = self.embeddings.embed_text(text)
                    if emb:
                        embeddings_map[node.id] = np.array(emb)
                except:
                    pass
            
            # Find similar pairs
            for i, node_a in enumerate(type_nodes):
                if node_a.id in processed:
                    continue
                
                similar_group = {node_a.id}
                
                for j in range(i + 1, len(type_nodes)):
                    node_b = type_nodes[j]
                    if node_b.id in processed:
                        continue
                    
                    # Check similarity
                    if self._are_similar(node_a, node_b, embeddings_map):
                        similar_group.add(node_b.id)
                
                if len(similar_group) >= self.min_occurrences_to_merge:
                    merge_groups.append(similar_group)
                    processed.update(similar_group)
        
        # Perform merges
        for group in merge_groups:
            if len(group) < 2:
                continue
            
            group_nodes = [memory.nodes[nid] for nid in group]
            merged_node, affected_edges = self._merge_nodes(group_nodes, memory)
            
            # Record event
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.CONSOLIDATION,
                memory_id=memory.id,
                affected_nodes=list(group),
                affected_edges=affected_edges,
                before_state={"node_count": len(group)},
                after_state={"merged_node": merged_node.id},
                reason=f"Merged {len(group)} similar entities into '{merged_node.name}'",
            ))
        
        return events
    
    def _are_similar(
        self,
        node_a: MemoryNode,
        node_b: MemoryNode,
        embeddings_map: Dict[str, np.ndarray],
    ) -> bool:
        """Check if two nodes are similar enough to merge."""
        # Name similarity
        name_a = node_a.canonical_name or node_a.name
        name_b = node_b.canonical_name or node_b.name
        
        # Check for alias overlap
        if node_a.aliases & node_b.aliases:
            return True
        
        # Check name containment
        if name_a.lower() in name_b.lower() or name_b.lower() in name_a.lower():
            return True
        
        # Check embedding similarity
        emb_a = embeddings_map.get(node_a.id)
        emb_b = embeddings_map.get(node_b.id)
        
        if emb_a is not None and emb_b is not None:
            similarity = self._cosine_similarity(emb_a, emb_b)
            if similarity >= self.similarity_threshold:
                return True
        
        return False
    
    def _merge_nodes(
        self,
        nodes: List[MemoryNode],
        memory: Memory,
    ) -> Tuple[MemoryNode, List[str]]:
        """Merge multiple nodes into one."""
        # Choose the best name (longest/most complete)
        best_node = max(nodes, key=lambda n: (len(n.name), n.access_count))
        
        # Collect all aliases
        all_aliases = set()
        all_descriptions = set()
        total_access = 0
        highest_importance = MemoryImportance.EPHEMERAL
        
        for node in nodes:
            all_aliases.update(node.aliases)
            all_aliases.add(node.name)
            if node.description:
                all_descriptions.add(node.description)
            total_access += node.access_count
            if node.importance.value > highest_importance.value:
                highest_importance = node.importance
        
        # Create merged node (preserving user_id for multi-tenant isolation)
        merged = MemoryNode(
            id=best_node.id,
            name=best_node.name,
            entity_type=best_node.entity_type,
            description=self._best_description(all_descriptions) or best_node.description,
            canonical_name=best_node.canonical_name or best_node.name,
            aliases=all_aliases,
            embedding=best_node.embedding,  # Preserve embedding
            properties={
                **best_node.properties,
                "merged_from": [n.id for n in nodes],
                "merge_count": len(nodes),
            },
            importance=highest_importance,
            access_count=total_access,
            user_id=best_node.user_id,  # Multi-tenant isolation
            memory_id=memory.id,
        )
        
        # Update edges to point to merged node
        affected_edges = []
        for node in nodes:
            if node.id == merged.id:
                continue
            
            for edge_id, edge in list(memory.edges.items()):
                updated = False
                new_source = edge.source_id
                new_target = edge.target_id
                
                if edge.source_id == node.id:
                    new_source = merged.id
                    updated = True
                if edge.target_id == node.id:
                    new_target = merged.id
                    updated = True
                
                if updated:
                    # Update edge
                    memory.edges[edge_id] = edge.evolve(
                        source_id=new_source,
                        target_id=new_target,
                    )
                    affected_edges.append(edge_id)
            
            # Remove merged node
            del memory.nodes[node.id]
        
        # Add/update merged node
        memory.nodes[merged.id] = merged
        
        return merged, affected_edges
    
    def _reinforce_edges(
        self,
        memory: Memory,
    ) -> List[EvolutionEvent]:
        """Strengthen edges that appear multiple times."""
        events = []
        
        # Group edges by (source, target, relation)
        edge_groups: Dict[Tuple[str, str, str], List[MemoryEdge]] = {}
        
        for edge in memory.edges.values():
            key = (edge.source_id, edge.target_id, edge.relation_type)
            edge_groups.setdefault(key, []).append(edge)
        
        # Merge duplicate edges
        for key, edges in edge_groups.items():
            if len(edges) < 2:
                continue
            
            # Keep strongest edge, reinforce it
            strongest = max(edges, key=lambda e: (e.weight, e.confidence))
            
            # Combine weights and confidence
            total_weight = sum(e.weight for e in edges)
            avg_confidence = sum(e.confidence for e in edges) / len(edges)
            
            reinforced = strongest.evolve(
                weight=min(10.0, total_weight),
                confidence=min(1.0, avg_confidence * 1.1),  # Slight boost
            )
            
            # Remove duplicates, keep reinforced
            for edge in edges:
                if edge.id != reinforced.id:
                    del memory.edges[edge.id]
            
            memory.edges[reinforced.id] = reinforced
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.REINFORCEMENT,
                memory_id=memory.id,
                affected_edges=[reinforced.id],
                before_state={"edge_count": len(edges)},
                after_state={"weight": reinforced.weight, "confidence": reinforced.confidence},
                reason=f"Reinforced edge from {len(edges)} occurrences",
            ))
        
        return events
    
    def _synthesize_knowledge(
        self,
        memory: Memory,
    ) -> List[EvolutionEvent]:
        """Create new knowledge by inferring from patterns."""
        events = []
        
        # Look for transitive relationships (A->B, B->C implies A->C)
        # This is a simplified version - could be much more sophisticated
        
        # Build adjacency
        outgoing: Dict[str, List[MemoryEdge]] = {}
        for edge in memory.edges.values():
            outgoing.setdefault(edge.source_id, []).append(edge)
        
        new_edges = []
        existing_pairs = {(e.source_id, e.target_id) for e in memory.edges.values()}
        
        for node_a_id in memory.nodes:
            edges_a = outgoing.get(node_a_id, [])
            
            for edge_ab in edges_a:
                node_b_id = edge_ab.target_id
                edges_b = outgoing.get(node_b_id, [])
                
                for edge_bc in edges_b:
                    node_c_id = edge_bc.target_id
                    
                    # Skip if A == C
                    if node_a_id == node_c_id:
                        continue
                    
                    # Skip if A->C already exists
                    if (node_a_id, node_c_id) in existing_pairs:
                        continue
                    
                    # Only infer if both edges are strong
                    if edge_ab.confidence < 0.7 or edge_bc.confidence < 0.7:
                        continue
                    
                    # Create inferred edge
                    new_edge = MemoryEdge(
                        id="",
                        source_id=node_a_id,
                        target_id=node_c_id,
                        relation_type="inferred_connection",
                        description=f"Inferred from {edge_ab.relation_type} and {edge_bc.relation_type}",
                        weight=min(edge_ab.weight, edge_bc.weight) * 0.5,
                        confidence=edge_ab.confidence * edge_bc.confidence * 0.8,
                        properties={
                            "inferred": True,
                            "via_node": node_b_id,
                            "source_edges": [edge_ab.id, edge_bc.id],
                        },
                        memory_id=memory.id,
                    )
                    
                    new_edges.append(new_edge)
                    existing_pairs.add((node_a_id, node_c_id))
        
        # Add synthesized edges
        for edge in new_edges[:10]:  # Limit to prevent explosion
            memory.add_edge(edge)
            
            events.append(EvolutionEvent(
                evolution_type=EvolutionType.SYNTHESIS,
                memory_id=memory.id,
                affected_edges=[edge.id],
                after_state={"edge_id": edge.id, "relation": edge.relation_type},
                reason="Synthesized transitive relationship",
            ))
        
        return events
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _best_description(self, descriptions: Set[str]) -> Optional[str]:
        """Choose the best description from a set."""
        if not descriptions:
            return None
        return max(descriptions, key=len)


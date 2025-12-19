"""
GraphMem Memory Consolidation

Merges related memories to create stronger, more coherent knowledge.
Like human memory consolidation during sleep.
"""

from __future__ import annotations
import logging
import re
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


def _get_importance_value(importance) -> float:
    """Safely get numeric value from importance (enum or float)."""
    if hasattr(importance, 'value'):
        return importance.value
    return float(importance) if importance is not None else 5.0


# LLM prompt for entity consolidation
ENTITY_CONSOLIDATION_PROMPT = """You are an expert at identifying when different names refer to the SAME entity.

## ENTITIES (same type: {entity_type})
{entities_list}

## TASK
Group entities that refer to the SAME real-world entity. Consider:
- "Dr. Chen", "Alexander Chen", "A. Chen", "Professor Chen" → SAME person
- "The Quantum Pioneer" could be a nickname for someone
- "Tesla, Inc.", "Tesla", "Tesla Motors" → SAME company
- Different descriptions may still be the same entity

## OUTPUT FORMAT
Output groups of entity IDs that should be merged. Each line is one group.
Format: ID1, ID2, ID3, ...

Example:
0, 2, 5
1, 3
4

If entity is unique (no matches), list it alone:
6

List ALL entities, either grouped or alone.

## GROUPS (one per line, entity IDs separated by commas):
"""


class MemoryConsolidation:
    """
    Consolidates similar memories into stronger, unified representations.
    
    Consolidation Types:
    1. Entity Merging: Merge duplicate/similar entities (LLM-based)
    2. Edge Strengthening: Strengthen frequently co-occurring relationships
    3. Cluster Refinement: Improve community summaries
    4. Knowledge Synthesis: Create new memories from patterns
    """
    
    def __init__(
        self,
        embeddings,
        llm=None,  # NEW: LLM for smart consolidation
        similarity_threshold: float = 0.85,
        min_occurrences_to_merge: int = 2,
        synthesis_enabled: bool = True,
    ):
        """
        Initialize consolidation handler.
        
        Args:
            embeddings: Embedding provider
            llm: LLM provider for smart entity matching
            similarity_threshold: Threshold for considering entities similar
            min_occurrences_to_merge: Minimum occurrences before merging
            synthesis_enabled: Whether to create synthesized memories
        """
        self.embeddings = embeddings
        self.llm = llm  # NEW: LLM for entity consolidation
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
        """Find and merge similar entities using LLM-based matching."""
        events = []
        
        nodes = list(memory.nodes.values())
        if len(nodes) < 2:
            return events
        
        # Group nodes by entity type for efficiency
        type_groups: Dict[str, List[MemoryNode]] = {}
        for node in nodes:
            key = node.entity_type.lower() if node.entity_type else "unknown"
            type_groups.setdefault(key, []).append(node)
        
        # Find merge candidates within each type using LLM - CONCURRENT
        all_merge_groups: List[Set[str]] = []
        
        # Process entity types in parallel for faster consolidation
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_entity_type(entity_type: str, type_nodes: List[MemoryNode]) -> List[Set[str]]:
            """Process single entity type (thread-safe)."""
            if len(type_nodes) < 2:
                return []
            
            # Use LLM to identify duplicates (if available)
            if self.llm and len(type_nodes) <= 50:  # Limit to avoid token overflow
                return self._llm_find_duplicates(type_nodes, entity_type)
            else:
                # Fallback to embedding-based matching for large sets
                return self._embedding_find_duplicates(type_nodes)
        
        # Use ThreadPool to process entity types concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_entity_type, entity_type, type_nodes): entity_type
                for entity_type, type_nodes in type_groups.items()
            }
            
            for future in as_completed(futures):
                entity_type = futures[future]
                try:
                    merge_groups = future.result()
                    all_merge_groups.extend(merge_groups)
                except Exception as e:
                    logger.warning(f"Entity type '{entity_type}' processing failed: {e}")
        
        # Perform merges
        processed_nodes = set()
        for group in all_merge_groups:
            if len(group) < 2:
                continue
            
            # Skip if any node already processed
            if any(nid in processed_nodes for nid in group):
                continue
            
            group_nodes = [memory.nodes[nid] for nid in group if nid in memory.nodes]
            if len(group_nodes) < 2:
                continue
            
            merged_node, affected_edges = self._merge_nodes(group_nodes, memory)
            processed_nodes.update(group)
            
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
            
            logger.info(f"🔗 Merged entities: {[n.name for n in group_nodes]} → '{merged_node.name}'")
        
        return events
    
    def _llm_find_duplicates(
        self,
        nodes: List[MemoryNode],
        entity_type: str,
    ) -> List[Set[str]]:
        """Use LLM to find duplicate entities."""
        if not self.llm:
            return []
        
        # Build entity list for prompt
        entities_list = []
        node_id_map = {}  # Map index to node ID
        for i, node in enumerate(nodes):
            node_id_map[i] = node.id
            aliases_str = ", ".join(node.aliases) if node.aliases else "none"
            desc = (node.description or "")[:100]
            entities_list.append(f"[{i}] {node.name} (aliases: {aliases_str}) - {desc}")
        
        prompt = ENTITY_CONSOLIDATION_PROMPT.format(
            entity_type=entity_type,
            entities_list="\n".join(entities_list),
        )
        
        try:
            response = self.llm.complete(prompt)
            
            # Parse response into groups
            merge_groups = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Parse comma-separated IDs
                try:
                    ids = [int(x.strip()) for x in line.split(",") if x.strip().isdigit()]
                    if len(ids) >= 2:
                        # Convert indices to node IDs
                        group = {node_id_map[i] for i in ids if i in node_id_map}
                        if len(group) >= 2:
                            merge_groups.append(group)
                            logger.debug(f"LLM group: {[nodes[i].name for i in ids if i < len(nodes)]}")
                except (ValueError, KeyError) as e:
                    continue
            
            logger.info(f"LLM found {len(merge_groups)} entity groups to merge in {entity_type}")
            return merge_groups
            
        except Exception as e:
            logger.warning(f"LLM consolidation failed: {e}, falling back to embeddings")
            return self._embedding_find_duplicates(nodes)
    
    def _embedding_find_duplicates(
        self,
        nodes: List[MemoryNode],
    ) -> List[Set[str]]:
        """Fallback: Find duplicates using embeddings."""
        merge_groups: List[Set[str]] = []
        processed: Set[str] = set()
        
        # Get embeddings
        embeddings_map = {}
        for node in nodes:
            text = node.description or node.name
            try:
                emb = self.embeddings.embed_text(text)
                if emb:
                    embeddings_map[node.id] = np.array(emb)
            except:
                pass
        
        # Find similar pairs
        for i, node_a in enumerate(nodes):
            if node_a.id in processed:
                continue
            
            similar_group = {node_a.id}
            
            for j in range(i + 1, len(nodes)):
                node_b = nodes[j]
                if node_b.id in processed:
                    continue
                
                if self._are_similar(node_a, node_b, embeddings_map):
                    similar_group.add(node_b.id)
            
            if len(similar_group) >= 2:
                merge_groups.append(similar_group)
                processed.update(similar_group)
        
        return merge_groups
    
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
        
        # CRITICAL: Check for EXACT name match (case-insensitive)
        if name_a.lower().strip() == name_b.lower().strip():
            return True
        
        # Check if either name matches the other's canonical name
        if node_a.name.lower().strip() == name_b.lower().strip():
            return True
        if node_b.name.lower().strip() == name_a.lower().strip():
            return True
        
        # Check for alias overlap (any alias matches)
        if node_a.aliases & node_b.aliases:
            return True
        
        # Check if any alias in A matches any name in B
        aliases_a_lower = {a.lower().strip() for a in node_a.aliases}
        aliases_b_lower = {b.lower().strip() for b in node_b.aliases}
        if aliases_a_lower & aliases_b_lower:
            return True
        
        # Check if name of one is in aliases of other
        if name_a.lower().strip() in aliases_b_lower:
            return True
        if name_b.lower().strip() in aliases_a_lower:
            return True
        
        # Check name containment
        if name_a.lower() in name_b.lower() or name_b.lower() in name_a.lower():
            return True
        
        # ===== NEW: AGGRESSIVE LAST NAME MATCHING =====
        # For person entities: "Dr. Chen" and "Alexander Chen" share "Chen"
        # This catches aliases where only the last name matches
        if self._share_last_name(name_a, name_b):
            # If they share last name AND have high description similarity, merge
            emb_a = embeddings_map.get(node_a.id)
            emb_b = embeddings_map.get(node_b.id)
            if emb_a is not None and emb_b is not None:
                sim = self._cosine_similarity(emb_a, emb_b)
                if sim >= 0.65:  # Lower threshold when names partially match
                    logger.debug(f"Last name match + embedding: '{name_a}' ≈ '{name_b}' (sim={sim:.2f})")
                    return True
        
        # ===== NEW: TOKEN OVERLAP =====
        # Check if significant name tokens overlap
        tokens_a = set(re.findall(r'\b[a-z]{2,}\b', name_a.lower()))
        tokens_b = set(re.findall(r'\b[a-z]{2,}\b', name_b.lower()))
        # Remove common words
        stopwords = {'the', 'dr', 'mr', 'ms', 'mrs', 'prof', 'professor', 'inc', 'corp', 'llc'}
        tokens_a -= stopwords
        tokens_b -= stopwords
        
        if tokens_a and tokens_b:
            overlap = len(tokens_a & tokens_b)
            min_tokens = min(len(tokens_a), len(tokens_b))
            if min_tokens > 0 and overlap / min_tokens >= 0.5:  # 50% token overlap
                # Check embedding to confirm
                emb_a = embeddings_map.get(node_a.id)
                emb_b = embeddings_map.get(node_b.id)
                if emb_a is not None and emb_b is not None:
                    sim = self._cosine_similarity(emb_a, emb_b)
                    if sim >= 0.60:  # Even lower threshold when tokens match
                        logger.debug(f"Token overlap + embedding: '{name_a}' ≈ '{name_b}' (sim={sim:.2f})")
                        return True
        
        # Check embedding similarity (lower threshold for same entity type)
        emb_a = embeddings_map.get(node_a.id)
        emb_b = embeddings_map.get(node_b.id)
        
        if emb_a is not None and emb_b is not None:
            similarity = self._cosine_similarity(emb_a, emb_b)
            # Lower threshold: 0.75 instead of 0.85
            if similarity >= 0.75:
                logger.debug(f"Embedding match: '{name_a}' ≈ '{name_b}' (sim={similarity:.2f})")
                return True
            
            # ===== LLM-BASED SIMILARITY CHECK (FINAL AUTHORITY) =====
            # If embedding similarity is moderate (0.5-0.75), ask the LLM
            if similarity >= 0.50 and self.llm is not None:
                if self._llm_confirms_same_entity(node_a, node_b):
                    return True
        
        return False
    
    def _llm_confirms_same_entity(
        self,
        node_a: MemoryNode,
        node_b: MemoryNode,
    ) -> bool:
        """Use LLM to determine if two entities are the same."""
        if self.llm is None:
            return False
        
        try:
            prompt = f"""You are an entity resolution expert. Determine if these two entities refer to the SAME real-world thing.

ENTITY A:
- Name: {node_a.name}
- Type: {node_a.entity_type}
- Aliases: {', '.join(node_a.aliases) if node_a.aliases else 'None'}
- Description: {node_a.description[:200] if node_a.description else 'None'}

ENTITY B:
- Name: {node_b.name}
- Type: {node_b.entity_type}
- Aliases: {', '.join(node_b.aliases) if node_b.aliases else 'None'}
- Description: {node_b.description[:200] if node_b.description else 'None'}

Consider:
- "Dr. Chen" and "Alexander Chen" could be the same person
- "SpaceX" and "Space Exploration Technologies Corp" are the same company
- Different people can share names if context differs

Answer ONLY with "YES" (same entity) or "NO" (different entities):"""

            response = self.llm.complete(prompt)
            answer = response.strip().upper()
            
            if "YES" in answer:
                logger.info(f"🤖 LLM confirmed: '{node_a.name}' = '{node_b.name}'")
                return True
            else:
                logger.debug(f"🤖 LLM rejected: '{node_a.name}' ≠ '{node_b.name}'")
                return False
                
        except Exception as e:
            logger.warning(f"LLM entity check failed: {e}")
            return False
    
    def _share_last_name(self, name_a: str, name_b: str) -> bool:
        """Check if two names share a last name (for person entities)."""
        # Extract potential last names (last word that's not a title)
        def get_last_name(name: str) -> Optional[str]:
            words = re.findall(r'\b[A-Za-z]{2,}\b', name)
            if not words:
                return None
            # Skip common prefixes/suffixes
            skip = {'dr', 'mr', 'ms', 'mrs', 'prof', 'professor', 'jr', 'sr', 'phd', 'md', 'the'}
            for word in reversed(words):
                if word.lower() not in skip:
                    return word.lower()
            return words[-1].lower() if words else None
        
        last_a = get_last_name(name_a)
        last_b = get_last_name(name_b)
        
        if last_a and last_b and last_a == last_b and len(last_a) >= 3:
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
            if _get_importance_value(node.importance) > _get_importance_value(highest_importance):
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


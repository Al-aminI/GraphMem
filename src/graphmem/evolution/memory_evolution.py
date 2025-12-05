"""
GraphMem Memory Evolution

Orchestrates all memory evolution operations:
- Consolidation (merging similar memories)
- Decay (forgetting less important memories)
- Rehydration (restoring relevant memories)
- Synthesis (creating new knowledge from patterns)

This is what makes GraphMem "self-improving" like human memory.

PERFORMANCE:
- Uses concurrent processing for LLM-based operations
- Parallel entity type processing during consolidation
- Batched LLM calls for decay analysis
"""

from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from graphmem.core.memory_types import (
    Memory,
    EvolutionEvent,
    EvolutionType,
)
from graphmem.evolution.consolidation import MemoryConsolidation
from graphmem.evolution.decay import MemoryDecay
from graphmem.evolution.rehydration import GraphRehydration
from graphmem.evolution.importance_scorer import ImportanceScorer

logger = logging.getLogger(__name__)


class MemoryEvolution:
    """
    Master controller for memory evolution.
    
    Memory evolution is what makes GraphMem different from static knowledge bases.
    Over time, the memory:
    - Consolidates related information for stronger, more coherent knowledge
    - Forgets less important details to prevent memory bloat
    - Strengthens frequently accessed memories
    - Synthesizes new knowledge from existing patterns
    
    This mimics how human memory works:
    - Sleep consolidation (merging experiences)
    - Forgetting curve (decay of unused memories)
    - Spaced repetition (strengthening through access)
    - Insight generation (connecting disparate knowledge)
    """
    
    def __init__(
        self,
        llm,
        embeddings,
        store,
        consolidation_threshold: float = 0.85,
        decay_enabled: bool = True,
        decay_half_life_days: float = 30.0,
        min_evolution_interval_hours: float = 1.0,
        max_workers: int = 10,  # NEW: Concurrent workers for LLM operations
    ):
        """
        Initialize evolution controller.
        
        Args:
            llm: LLM provider
            embeddings: Embedding provider
            store: Graph store
            consolidation_threshold: Similarity threshold for merging
            decay_enabled: Whether to enable memory decay
            decay_half_life_days: Half-life for memory decay
            min_evolution_interval_hours: Minimum time between evolutions
            max_workers: Number of concurrent workers for LLM operations
        """
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.min_evolution_interval = timedelta(hours=min_evolution_interval_hours)
        self.max_workers = max_workers
        
        # Initialize sub-components
        self.consolidation = MemoryConsolidation(
            embeddings=embeddings,
            llm=llm,  # Pass LLM for smart entity consolidation
            similarity_threshold=consolidation_threshold,
        )
        
        self.decay = MemoryDecay(
            llm=llm,  # LLM for intelligent decay reasoning
            half_life_days=decay_half_life_days,
        ) if decay_enabled else None
        
        self.rehydration = GraphRehydration(
            embeddings=embeddings,
        )
        
        self.importance_scorer = ImportanceScorer()
        
        # Track evolution state
        self._last_evolution: Dict[str, datetime] = {}
    
    def evolve(
        self,
        memory: Memory,
        evolution_types: Optional[List[EvolutionType]] = None,
        force: bool = False,
    ) -> List[EvolutionEvent]:
        """
        Evolve the memory.
        
        Args:
            memory: Memory to evolve
            evolution_types: Specific types to run. If None, runs all.
            force: If True, runs even if recently evolved.
        
        Returns:
            List of evolution events
        """
        # Check if we should evolve
        if not force:
            last_evolved = self._last_evolution.get(memory.id)
            if last_evolved:
                since_last = datetime.utcnow() - last_evolved
                if since_last < self.min_evolution_interval:
                    logger.debug(f"Skipping evolution - last evolved {since_last} ago")
                    return []
        
        all_events = []
        start_time = time.time()
        
        # Determine which evolution types to run
        if evolution_types is None:
            evolution_types = [
                EvolutionType.CONSOLIDATION,
                EvolutionType.DECAY,
                EvolutionType.REINFORCEMENT,
            ]
        
        # CONCURRENT EVOLUTION - run consolidation and decay in parallel
        # Both are LLM-intensive but independent operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit consolidation task
            if EvolutionType.CONSOLIDATION in evolution_types:
                futures[executor.submit(self._run_consolidation, memory)] = "consolidation"
            
            # Submit decay task
            if EvolutionType.DECAY in evolution_types and self.decay:
                futures[executor.submit(self._run_decay, memory)] = "decay"
            
            # Collect results
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    events = future.result()
                    all_events.extend(events)
                    logger.info(f"{task_name.title()}: {len(events)} events")
                except Exception as e:
                    logger.error(f"{task_name.title()} failed: {e}")
        
        # Update importance scores
        if EvolutionType.REINFORCEMENT in evolution_types:
            try:
                events = self._update_importance_scores(memory)
                all_events.extend(events)
            except Exception as e:
                logger.error(f"Importance update failed: {e}")
        
        # Record evolution time
        self._last_evolution[memory.id] = datetime.utcnow()
        
        # Update memory version
        memory.version += 1
        memory.updated_at = datetime.utcnow()
        
        elapsed = time.time() - start_time
        logger.info(f"Evolution complete: {len(all_events)} total events in {elapsed:.1f}s")
        return all_events
    
    def _run_consolidation(self, memory: Memory) -> List[EvolutionEvent]:
        """Run consolidation (thread-safe wrapper)."""
        return self.consolidation.consolidate(memory)
    
    def _run_decay(self, memory: Memory) -> List[EvolutionEvent]:
        """Run decay (thread-safe wrapper)."""
        return self.decay.apply_decay(memory)
    
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
            context: Context for relevance matching
            max_nodes: Maximum nodes to rehydrate
        
        Returns:
            Rehydration statistics
        """
        return self.rehydration.rehydrate(
            memory=memory,
            context=context,
            max_nodes=max_nodes,
        )
    
    def _update_importance_scores(
        self,
        memory: Memory,
    ) -> List[EvolutionEvent]:
        """Update importance scores for all elements."""
        events = []
        
        all_edges = list(memory.edges.values())
        all_nodes = list(memory.nodes.values())
        
        for node_id, node in memory.nodes.items():
            new_importance = self.importance_scorer.update_importance(
                node, all_edges, all_nodes
            )
            
            if new_importance != node.importance:
                old_importance = node.importance
                memory.nodes[node_id] = node.evolve(importance=new_importance)
                
                # Only track significant changes
                if abs(new_importance.value - old_importance.value) >= 2:
                    events.append(EvolutionEvent(
                        evolution_type=EvolutionType.REINFORCEMENT,
                        memory_id=memory.id,
                        affected_nodes=[node_id],
                        before_state={"importance": old_importance.name},
                        after_state={"importance": new_importance.name},
                        reason="Importance score updated",
                    ))
        
        return events
    
    def get_evolution_stats(
        self,
        memory: Memory,
    ) -> Dict[str, Any]:
        """Get statistics about memory evolution."""
        return {
            "memory_id": memory.id,
            "version": memory.version,
            "last_evolved": self._last_evolution.get(memory.id, memory.created_at).isoformat(),
            "total_nodes": len(memory.nodes),
            "total_edges": len(memory.edges),
            "total_clusters": len(memory.clusters),
            "active_nodes": sum(1 for n in memory.nodes.values() if n.state.name == "ACTIVE"),
            "archived_nodes": sum(1 for n in memory.nodes.values() if n.state.name == "ARCHIVED"),
            "deleted_nodes": sum(1 for n in memory.nodes.values() if n.state.name == "DELETED"),
            "importance_distribution": self._get_importance_distribution(memory),
        }
    
    def _get_importance_distribution(
        self,
        memory: Memory,
    ) -> Dict[str, int]:
        """Get distribution of importance levels."""
        distribution = {}
        for node in memory.nodes.values():
            level = node.importance.name
            distribution[level] = distribution.get(level, 0) + 1
        return distribution


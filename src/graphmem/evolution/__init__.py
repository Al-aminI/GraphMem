"""
GraphMem Evolution Module

Self-improving memory capabilities including consolidation, decay, and rehydration.
This makes GraphMem work like human memory - strengthening important memories
and letting less important ones fade over time.
"""

from graphmem.evolution.memory_evolution import MemoryEvolution
from graphmem.evolution.consolidation import MemoryConsolidation
from graphmem.evolution.decay import MemoryDecay
from graphmem.evolution.rehydration import GraphRehydration
from graphmem.evolution.importance_scorer import ImportanceScorer

__all__ = [
    "MemoryEvolution",
    "MemoryConsolidation",
    "MemoryDecay",
    "GraphRehydration",
    "ImportanceScorer",
]


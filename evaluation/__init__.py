"""
GraphMem Evaluation Module

Comprehensive benchmarks for evaluating GraphMem against state-of-the-art
agent memory systems.

Includes:
- Core benchmarks (accuracy, latency, scalability)
- Context engineering evaluation
- SOTA system comparisons
"""

from .benchmarks import (
    BenchmarkResult,
    EvaluationDataset,
    MemoryEvaluator,
    BenchmarkRunner,
    NaiveRAGBaseline,
    BufferMemoryBaseline,
    generate_entity_resolution_dataset,
    generate_multihop_reasoning_dataset,
    generate_temporal_memory_dataset,
    generate_long_context_dataset,
    run_graphmem_evaluation,
)

from .context_engineering import (
    ContextChunk,
    ContextWindow,
    ChunkingStrategy,
    ContextEngineeringEvaluator,
    run_context_engineering_evaluation,
)

from .sota_comparison import (
    SOTASystem,
    SOTA_SYSTEMS,
    SOTAEvaluator,
    run_full_evaluation,
)

__all__ = [
    # Core benchmarks
    "BenchmarkResult",
    "EvaluationDataset",
    "MemoryEvaluator",
    "BenchmarkRunner",
    "NaiveRAGBaseline",
    "BufferMemoryBaseline",
    "generate_entity_resolution_dataset",
    "generate_multihop_reasoning_dataset",
    "generate_temporal_memory_dataset",
    "generate_long_context_dataset",
    "run_graphmem_evaluation",
    # Context engineering
    "ContextChunk",
    "ContextWindow",
    "ChunkingStrategy",
    "ContextEngineeringEvaluator",
    "run_context_engineering_evaluation",
    # SOTA comparison
    "SOTASystem",
    "SOTA_SYSTEMS",
    "SOTAEvaluator",
    "run_full_evaluation",
]


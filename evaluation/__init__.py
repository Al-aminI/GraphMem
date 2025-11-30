"""
GraphMem Evaluation Module

Comprehensive benchmarks for evaluating GraphMem against state-of-the-art
agent memory systems.
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

__all__ = [
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
]


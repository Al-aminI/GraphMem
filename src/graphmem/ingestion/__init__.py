"""
High-Performance Ingestion Pipeline
====================================

State-of-the-art concurrent and distributed ingestion for GraphMem.

Features:
- Batch embedding generation (10x faster)
- Async LLM extraction with rate limiting
- Worker pool architecture (ThreadPool + AsyncIO)
- Streaming pipeline with backpressure
- Optional Redpanda/Kafka integration for distributed workloads

Usage:
    from graphmem.ingestion import HighPerformancePipeline
    
    pipeline = HighPerformancePipeline(
        llm=llm,
        embeddings=embeddings,
        max_workers=16,
        batch_size=50,
    )
    
    # Ingest documents concurrently
    results = pipeline.ingest_batch(documents)
    
    # Or stream from Redpanda/Kafka
    pipeline.stream_from_kafka(topic="documents")
"""

from graphmem.ingestion.pipeline import (
    HighPerformancePipeline,
    PipelineConfig,
    IngestResult,
)

from graphmem.ingestion.batch_embedder import (
    BatchEmbedder,
    EmbeddingBatch,
)

from graphmem.ingestion.async_extractor import (
    AsyncExtractor,
    ExtractionResult,
)

from graphmem.ingestion.auto_scale import (
    AutoScaler,
    HardwareInfo,
    OptimalConfig,
    get_optimal_workers,
    detect_and_configure,
    GPUOptimizer,
)

from graphmem.ingestion.benchmark import (
    Benchmark,
    BenchmarkResult,
    ComparisonResult,
    quick_benchmark,
)

__all__ = [
    "HighPerformancePipeline",
    "PipelineConfig", 
    "IngestResult",
    "BatchEmbedder",
    "EmbeddingBatch",
    "AsyncExtractor",
    "ExtractionResult",
    "AutoScaler",
    "HardwareInfo",
    "OptimalConfig",
    "get_optimal_workers",
    "detect_and_configure",
    "GPUOptimizer",
    "Benchmark",
    "BenchmarkResult",
    "ComparisonResult",
    "quick_benchmark",
]


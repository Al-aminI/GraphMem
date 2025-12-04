"""
Benchmark Utilities
===================

Measure actual speedups for different configurations.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    mode: str
    workers: int
    documents: int
    total_time: float
    throughput: float  # docs/sec
    avg_time_per_doc: float
    entities_extracted: int
    relationships_extracted: int


@dataclass
class ComparisonResult:
    """Comparison between sequential and parallel modes."""
    sequential: BenchmarkResult
    parallel: BenchmarkResult
    speedup: float  # parallel vs sequential


class Benchmark:
    """Benchmark GraphMem ingestion performance."""
    
    def __init__(self, memory):
        self.memory = memory
        self.results: List[BenchmarkResult] = []
    
    def run(
        self,
        documents: List[Dict[str, Any]],
        worker_counts: Optional[List[int]] = None,
    ) -> ComparisonResult:
        """Run benchmark comparing sequential vs parallel ingestion."""
        if len(documents) < 2:
            raise ValueError("Need at least 2 documents")
        
        if worker_counts is None:
            try:
                from graphmem.ingestion import get_optimal_workers
                optimal = get_optimal_workers(self.memory.config.llm_provider)
                parallel_workers = optimal.get("extraction_workers", 5)
            except:
                parallel_workers = 5
            worker_counts = [1, parallel_workers]
        
        logger.info(f"Running benchmark with {len(documents)} documents")
        
        results = []
        for workers in worker_counts:
            result = self._run_single(documents, workers)
            results.append(result)
            self.results.append(result)
        
        sequential = results[0]
        parallel = results[-1]
        speedup = sequential.total_time / parallel.total_time if parallel.total_time > 0 else 1.0
        
        return ComparisonResult(sequential=sequential, parallel=parallel, speedup=speedup)
    
    def _run_single(self, documents: List[Dict[str, Any]], workers: int) -> BenchmarkResult:
        """Run a single benchmark."""
        try:
            self.memory.clear()
        except:
            pass
        
        start_time = time.perf_counter()
        
        result = self.memory.ingest_batch(
            documents,
            max_workers=workers,
            show_progress=False,
            auto_scale=False,
        )
        
        total_time = time.perf_counter() - start_time
        docs_processed = result.get("documents_processed", len(documents))
        throughput = docs_processed / total_time if total_time > 0 else 0
        
        return BenchmarkResult(
            mode="parallel" if workers > 1 else "sequential",
            workers=workers,
            documents=docs_processed,
            total_time=total_time,
            throughput=throughput,
            avg_time_per_doc=total_time / docs_processed if docs_processed > 0 else 0,
            entities_extracted=result.get("total_entities", 0),
            relationships_extracted=result.get("total_relationships", 0),
        )
    
    def print_results(self):
        """Print benchmark results."""
        if not self.results:
            print("No results yet.")
            return
        
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Workers':>8} | {'Time (s)':>10} | {'Docs/sec':>10} | {'Entities':>10}")
        print("-" * 70)
        
        for r in self.results:
            print(f"{r.workers:>8} | {r.total_time:>10.2f} | {r.throughput:>10.3f} | {r.entities_extracted:>10}")
        
        if len(self.results) > 1 and self.results[0].workers == 1:
            baseline = self.results[0].total_time
            print("-" * 70)
            for r in self.results[1:]:
                speedup = baseline / r.total_time if r.total_time > 0 else 1.0
                print(f"  {r.workers} workers: {speedup:.2f}x speedup")
        print("=" * 70)


def quick_benchmark(memory, n_docs: int = 5) -> ComparisonResult:
    """Quick benchmark with synthetic documents."""
    test_docs = []
    for i in range(n_docs):
        content = f"Company {i} founded in 20{i:02d}. CEO John Smith{i}. HQ in City {i}."
        test_docs.append({"id": f"bench_{i}", "content": content})
    
    bench = Benchmark(memory)
    return bench.run(test_docs)

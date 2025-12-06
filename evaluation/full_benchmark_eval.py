# =============================================================================
# 📊 GraphMem - FULL MEMORYAGENTBENCH EVALUATION
# 
# Runs all four competencies from the MemoryAgentBench benchmark:
# 1. AR  - Accurate Retrieval
# 2. TTL - Test Time Learning  
# 3. LRU - Long Range Understanding
# 4. SF  - Semantic Filtering (Conflict Resolution)
#
# Produces results in Table 2 format from the paper.
# =============================================================================

import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

# Import individual evaluators (use relative imports for package, absolute for direct run)
try:
    from .accurate_retrieval_eval import (
        evaluate_accurate_retrieval,
        print_accurate_retrieval_results,
        split_ar_context,
    )
    from .test_time_learning_eval import (
        evaluate_test_time_learning,
        print_test_time_learning_results,
        split_ttl_context,
    )
    from .long_range_understanding_eval import (
        evaluate_long_range_understanding,
        print_long_range_understanding_results,
        split_lru_context,
    )
    from .conflict_resolution_eval import (
        evaluate_conflict_resolution,
        print_conflict_resolution_results,
        split_sf_context,
    )
except ImportError:
    # Fallback for direct script execution
    from accurate_retrieval_eval import (
        evaluate_accurate_retrieval,
        print_accurate_retrieval_results,
        split_ar_context,
    )
    from test_time_learning_eval import (
        evaluate_test_time_learning,
        print_test_time_learning_results,
        split_ttl_context,
    )
    from long_range_understanding_eval import (
        evaluate_long_range_understanding,
        print_long_range_understanding_results,
        split_lru_context,
    )
    from conflict_resolution_eval import (
        evaluate_conflict_resolution,
        print_conflict_resolution_results,
        split_sf_context,
    )

# =============================================================================
# RESULT AGGREGATION
# =============================================================================

@dataclass
class FullBenchmarkResults:
    """Aggregated results across all competencies."""
    timestamp: str = ""
    
    # Per-competency metrics
    ar_metrics: Dict[str, float] = field(default_factory=dict)
    ttl_metrics: Dict[str, float] = field(default_factory=dict)
    lru_metrics: Dict[str, float] = field(default_factory=dict)
    sf_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Overall score
    overall_score: float = 0.0
    
    # Timing
    total_time_seconds: float = 0.0


def run_full_benchmark(
    config,  # MemoryConfig for creating fresh GraphMem instances per sample
    # AR settings
    ar_max_samples: int = 10,
    ar_max_questions: int = 10,
    # TTL settings
    ttl_max_samples: int = 5,
    ttl_max_questions: int = 10,
    # LRU settings
    lru_max_samples: int = 3,
    lru_max_questions: int = 5,
    # SF settings
    sf_max_samples: int = 3,
    sf_max_questions: int = 30,
    # General settings
    max_concurrent: int = 5,
    show_details: bool = False,
    run_evolution: bool = True,
    save_results: bool = True,
    turso_db_dir: str = "eval_dbs",
) -> FullBenchmarkResults:
    """
    Run the full MemoryAgentBench benchmark.
    
    EACH SAMPLE GETS A FRESH GRAPHMEM INSTANCE WITH LOCAL TURSO DB.
    
    Args:
        config: MemoryConfig for creating fresh GraphMem instances per sample
        ar_max_samples: Max samples for Accurate Retrieval
        ar_max_questions: Max questions per AR sample
        ttl_max_samples: Max samples for Test Time Learning
        ttl_max_questions: Max questions per TTL sample
        lru_max_samples: Max samples for Long Range Understanding
        lru_max_questions: Max questions per LRU sample
        sf_max_samples: Max samples for Conflict Resolution
        sf_max_questions: Max questions per SF sample
        max_concurrent: Concurrent query workers
        show_details: Print per-question details
        run_evolution: Run evolve() after ingestion
        save_results: Save results to JSON file
        turso_db_dir: Directory for Turso database files
    
    Returns:
        FullBenchmarkResults object
    """
    import os
    
    # Create directory for Turso DBs
    os.makedirs(turso_db_dir, exist_ok=True)
    
    results = FullBenchmarkResults()
    results.timestamp = datetime.now().isoformat()
    
    total_start = time.time()
    
    print("\n" + "="*80)
    print("🧠 GRAPHMEM FULL BENCHMARK - MEMORYAGENTBENCH")
    print("="*80)
    print(f"\n⏰ Started at: {results.timestamp}")
    print(f"📊 Testing all 4 competencies: AR, TTL, LRU, SF")
    print(f"🗄️ Each sample gets fresh Turso DB in: {turso_db_dir}/")
    print("="*80)
    
    # =========================================================================
    # 1. ACCURATE RETRIEVAL (AR)
    # =========================================================================
    print("\n\n" + "🔹"*40)
    print("📋 COMPETENCY 1/4: ACCURATE RETRIEVAL (AR)")
    print("🔹"*40)
    
    try:
        ar_metrics, ar_results = evaluate_accurate_retrieval(
            config=config,
            max_samples=ar_max_samples,
            max_questions_per_sample=ar_max_questions,
            max_concurrent=max_concurrent,
            show_details=show_details,
            run_evolution=run_evolution,
            turso_db_prefix=f"{turso_db_dir}/ar",
        )
        results.ar_metrics = ar_metrics
        print_accurate_retrieval_results(ar_metrics)
    except Exception as e:
        print(f"❌ AR evaluation failed: {e}")
        results.ar_metrics = {"error": str(e)}
    
    # =========================================================================
    # 2. TEST TIME LEARNING (TTL)
    # =========================================================================
    print("\n\n" + "🔹"*40)
    print("📋 COMPETENCY 2/4: TEST TIME LEARNING (TTL)")
    print("🔹"*40)
    
    try:
        ttl_metrics, ttl_results = evaluate_test_time_learning(
            config=config,
            max_samples=ttl_max_samples,
            max_questions_per_sample=ttl_max_questions,
            max_concurrent=max_concurrent,
            show_details=show_details,
            run_evolution=run_evolution,
            turso_db_prefix=f"{turso_db_dir}/ttl",
        )
        results.ttl_metrics = ttl_metrics
        print_test_time_learning_results(ttl_metrics)
    except Exception as e:
        print(f"❌ TTL evaluation failed: {e}")
        results.ttl_metrics = {"error": str(e)}
    
    # =========================================================================
    # 3. LONG RANGE UNDERSTANDING (LRU)
    # =========================================================================
    print("\n\n" + "🔹"*40)
    print("📋 COMPETENCY 3/4: LONG RANGE UNDERSTANDING (LRU)")
    print("🔹"*40)
    
    try:
        lru_metrics, lru_results = evaluate_long_range_understanding(
            config=config,
            max_samples=lru_max_samples,
            max_questions_per_sample=lru_max_questions,
            max_concurrent=max_concurrent,
            show_details=show_details,
            run_evolution=run_evolution,
            turso_db_prefix=f"{turso_db_dir}/lru",
        )
        results.lru_metrics = lru_metrics
        print_long_range_understanding_results(lru_metrics)
    except Exception as e:
        print(f"❌ LRU evaluation failed: {e}")
        results.lru_metrics = {"error": str(e)}
    
    # =========================================================================
    # 4. CONFLICT RESOLUTION (SF)
    # =========================================================================
    print("\n\n" + "🔹"*40)
    print("📋 COMPETENCY 4/4: CONFLICT RESOLUTION (SF)")
    print("🔹"*40)
    
    try:
        sf_metrics, sf_results = evaluate_conflict_resolution(
            config=config,
            max_samples=sf_max_samples,
            max_questions_per_sample=sf_max_questions,
            max_concurrent=max_concurrent,
            show_details=show_details,
            run_evolution=run_evolution,
            turso_db_prefix=f"{turso_db_dir}/sf",
        )
        results.sf_metrics = sf_metrics
        print_conflict_resolution_results(sf_metrics)
    except Exception as e:
        print(f"❌ SF evaluation failed: {e}")
        results.sf_metrics = {"error": str(e)}
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    results.total_time_seconds = time.time() - total_start
    
    # Calculate overall score (average across competencies)
    scores = []
    if 'ar_avg' in results.ar_metrics:
        scores.append(results.ar_metrics['ar_avg'])
    elif 'substring_match' in results.ar_metrics:
        scores.append(results.ar_metrics['substring_match'])
    
    if 'ttl_avg' in results.ttl_metrics:
        scores.append(results.ttl_metrics['ttl_avg'])
    elif 'substring_match' in results.ttl_metrics:
        scores.append(results.ttl_metrics['substring_match'])
    
    if 'lru_avg' in results.lru_metrics:
        scores.append(results.lru_metrics['lru_avg'])
    elif 'substring_match' in results.lru_metrics:
        scores.append(results.lru_metrics['substring_match'])
    
    if 'substring_match' in results.sf_metrics:
        scores.append(results.sf_metrics['substring_match'])
    
    results.overall_score = np.mean(scores) if scores else 0.0
    
    # Print final summary
    print_full_table2(results)
    
    # Save results
    if save_results:
        save_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(save_path, 'w') as f:
            json.dump({
                "timestamp": results.timestamp,
                "ar_metrics": results.ar_metrics,
                "ttl_metrics": results.ttl_metrics,
                "lru_metrics": results.lru_metrics,
                "sf_metrics": results.sf_metrics,
                "overall_score": results.overall_score,
                "total_time_seconds": results.total_time_seconds,
            }, f, indent=2)
        print(f"\n💾 Results saved to: {save_path}")
    
    return results


def print_full_table2(results: FullBenchmarkResults):
    """Print complete Table 2 with all competencies."""
    
    print("\n\n" + "="*100)
    print("📊 MEMORYAGENTBENCH - COMPLETE TABLE 2 RESULTS")
    print("="*100)
    
    # Get scores
    ar_avg = results.ar_metrics.get('ar_avg', results.ar_metrics.get('substring_match', 0))
    ttl_avg = results.ttl_metrics.get('ttl_avg', results.ttl_metrics.get('substring_match', 0))
    lru_avg = results.lru_metrics.get('lru_avg', results.lru_metrics.get('substring_match', 0))
    sf_avg = results.sf_metrics.get('substring_match', 0)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    MEMORYAGENTBENCH TABLE 2                                      │
├──────────────────────┬──────────────────┬──────────────────┬──────────────────┬─────────────────┤
│      Agent Type      │   AR (Accurate   │  TTL (Test-Time  │  LRU (Long-Range │  SF (Conflict   │
│                      │    Retrieval)    │    Learning)     │  Understanding)  │   Resolution)   │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ GPT-4o-mini          │      49.2        │       48.6       │       46.2       │      25.0       │
│ GPT-4.1-mini         │      71.8        │       46.2       │       49.1       │      20.5       │
│ Text-Embed-3-Small   │      53.8        │       42.7       │       36.3       │      15.5       │
│ GraphRAG             │      40.9        │       24.8       │       19.9       │       8.0       │
│ HippoRAG-v2          │      65.1        │       35.8       │       36.2       │      29.5       │
│ Mem0                 │      32.6        │       21.2       │       20.7       │      10.0       │
├──────────────────────┼──────────────────┼──────────────────┼──────────────────┼─────────────────┤
│ 🧠 GRAPHMEM (OURS)   │     {ar_avg:5.1f}        │      {ttl_avg:5.1f}       │      {lru_avg:5.1f}       │     {sf_avg:5.1f}       │
└──────────────────────┴──────────────────┴──────────────────┴──────────────────┴─────────────────┘
""")
    
    # Calculate overall comparison
    our_overall = results.overall_score
    
    baselines = {
        "GPT-4o-mini": (49.2 + 48.6 + 46.2 + 25.0) / 4,  # 42.25
        "GPT-4.1-mini": (71.8 + 46.2 + 49.1 + 20.5) / 4,  # 46.9
        "Text-Embed-3-Small": (53.8 + 42.7 + 36.3 + 15.5) / 4,  # 37.08
        "GraphRAG": (40.9 + 24.8 + 19.9 + 8.0) / 4,  # 23.4
        "HippoRAG-v2": (65.1 + 35.8 + 36.2 + 29.5) / 4,  # 41.65
        "Mem0": (32.6 + 21.2 + 20.7 + 10.0) / 4,  # 21.13
    }
    
    print(f"\n📈 OVERALL COMPARISON:")
    print(f"   GraphMem Overall Score: {our_overall:.1f}%")
    print()
    
    beaten = []
    for name, score in sorted(baselines.items(), key=lambda x: x[1], reverse=True):
        status = "🏆" if our_overall > score else "  "
        print(f"   {status} {name}: {score:.1f}%")
        if our_overall > score:
            beaten.append(name)
    
    print()
    if beaten:
        print(f"✅ GraphMem BEATS: {', '.join(beaten)}")
    
    print(f"\n⏱️  Total benchmark time: {results.total_time_seconds:.1f} seconds")
    print(f"📅 Completed at: {datetime.now().isoformat()}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║              FULL MEMORYAGENTBENCH BENCHMARK FOR GRAPHMEM                  ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║                                                                             ║
    ║  This runs all 4 competencies:                                             ║
    ║                                                                             ║
    ║  1. AR  - Accurate Retrieval (SH-QA, MH-QA, LME, EventQA)                  ║
    ║  2. TTL - Test Time Learning (MCC, Recommendations)                        ║
    ║  3. LRU - Long Range Understanding (Summarization, DetQA)                  ║
    ║  4. SF  - Conflict Resolution (Fact Consolidation)                         ║
    ║                                                                             ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    To run in a notebook:
    
    from graphmem import MemoryConfig
    from full_benchmark_eval import run_full_benchmark
    
    # Create config (each sample will get its own fresh GraphMem + Turso DB)
    config = MemoryConfig(
        llm_provider="azure",
        llm_api_key="YOUR_KEY",
        llm_api_base="https://YOUR-ENDPOINT.openai.azure.com/",
        llm_model="gpt-4.1-mini",
        embedding_provider="azure",
        embedding_api_key="YOUR_KEY",
        embedding_api_base="https://YOUR-ENDPOINT.openai.azure.com/",
        embedding_model="text-embedding-3-small",
    )
    
    # Run full benchmark - each sample gets fresh GraphMem + local Turso DB
    results = run_full_benchmark(
        config,  # Pass config, not memory!
        ar_max_samples=10,
        ar_max_questions=10,
        ttl_max_samples=5,
        lru_max_samples=3,
        sf_max_samples=3,
        sf_max_questions=30,
        max_concurrent=5,
        show_details=False,
        run_evolution=True,  # CRITICAL!
        save_results=True,
        turso_db_dir="eval_dbs",  # Directory for per-sample Turso DBs
    )
    """)


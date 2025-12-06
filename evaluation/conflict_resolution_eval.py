# =============================================================================
# 📊 GraphMem - CONFLICT RESOLUTION (SF) Evaluation
# 
# Tests whether GraphMem correctly prefers NEWER facts over OLDER ones
# when there are conflicts in the knowledge base.
# =============================================================================

import time
import string
import re
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datasets import load_dataset

# =============================================================================
# CONTEXT SPLITTING - SF uses numbered facts: "0. fact", "1. fact", etc.
# =============================================================================

def split_sf_context(text: str) -> List[Dict[str, str]]:
    """
    Split SF (Conflict Resolution) context into individual facts.
    Format: "0. Thomas Kyd was born in London.\n1. The chairperson..."
    """
    # Split on: number + dot + optional spaces
    parts = re.split(r'\d+\.\s*', text)
    # First item before "0." is header text, skip if empty
    parts = [p.strip() for p in parts if p.strip()]
    
    # Convert to {"content": "..."} format with fact number for priority
    documents = []
    for i, part in enumerate(parts):
        if part:
            documents.append({
                "content": f"Fact {i}: {part}",
                "metadata": {"fact_number": i, "priority": i}  # Higher = newer
            })
    
    return documents

# =============================================================================
# METRICS
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize text for evaluation."""
    text = str(text).lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(truth_tokens) if truth_tokens else 0
    
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def substring_match(prediction: str, ground_truth: str) -> float:
    """Check if ground truth is in prediction."""
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))

def calculate_metrics(prediction: str, ground_truths: Any) -> Dict[str, float]:
    """Calculate all metrics for a prediction."""
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif isinstance(ground_truths, list):
        gt_list = [str(g) for g in ground_truths]
    else:
        gt_list = [str(ground_truths)]
    
    return {
        "f1": max(f1_score(prediction, gt) for gt in gt_list),
        "substring_match": max(substring_match(prediction, gt) for gt in gt_list),
    }

# =============================================================================
# RESULT STORAGE
# =============================================================================

@dataclass
class QueryResult:
    query: str
    expected: str
    predicted: str
    f1: float = 0.0
    substring_match: float = 0.0
    latency_ms: float = 0.0

# =============================================================================
# CONFLICT RESOLUTION EVALUATION
# =============================================================================

def evaluate_conflict_resolution(
    config,  # MemoryConfig for creating fresh instances
    max_samples: int = 3,
    max_questions_per_sample: int = 20,
    max_concurrent: int = 5,
    show_details: bool = True,
    run_evolution: bool = True,
    turso_db_prefix: str = "sf_eval",
) -> Dict[str, Any]:
    """
    Evaluate GraphMem on Conflict Resolution task.
    
    The key insight: Facts are numbered, and LATER facts should override EARLIER ones.
    GraphMem's decay mechanism should help identify and prioritize newer information.
    
    EACH SAMPLE GETS A FRESH GRAPHMEM INSTANCE WITH LOCAL TURSO DB.
    
    Args:
        config: MemoryConfig for creating fresh GraphMem instances
        max_samples: Number of samples to test
        max_questions_per_sample: Questions per sample
        max_concurrent: Concurrent queries
        show_details: Print details
        run_evolution: Whether to run evolve() after ingestion
        turso_db_prefix: Prefix for local Turso database files
    
    Returns:
        Dictionary with metrics
    """
    from graphmem import GraphMem
    import os
    
    print("📥 Loading Conflict_Resolution dataset...")
    ds = load_dataset('ai-hyz/MemoryAgentBench')
    cr = ds['Conflict_Resolution']
    print(f"   Found {len(cr)} samples with conflicts\n")
    
    all_results = []
    
    for sample_idx in range(min(max_samples, len(cr))):
        sample = cr[sample_idx]
        context = sample.get('context', '')
        questions = sample.get('questions', [])
        answers = sample.get('answers', [])
        
        # === CREATE FRESH GRAPHMEM WITH LOCAL TURSO DB ===
        db_path = f"{turso_db_prefix}_sample_{sample_idx}.db"
        
        # Remove old DB if exists for fresh start
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(db_path + "_cache.db"):
            os.remove(db_path + "_cache.db")
        
        # Create new config with Turso storage
        from graphmem import MemoryConfig
        sample_config = MemoryConfig(
            # LLM settings from original config
            llm_provider=config.llm_provider,
            llm_api_key=config.llm_api_key,
            llm_api_base=config.llm_api_base,
            llm_model=config.llm_model,
            # Embedding settings from original config
            embedding_provider=config.embedding_provider,
            embedding_api_key=config.embedding_api_key,
            embedding_api_base=config.embedding_api_base,
            embedding_model=config.embedding_model,
            # Azure settings if present
            azure_api_version=getattr(config, 'azure_api_version', "2024-12-01-preview"),
            azure_deployment=getattr(config, 'azure_deployment', None),
            azure_embedding_deployment=getattr(config, 'azure_embedding_deployment', None),
            # Use local Turso for isolated storage
            turso_db_path=db_path,
        )
        
        memory = GraphMem(sample_config)
        print(f"\n🗄️ Created fresh GraphMem with Turso DB: {db_path}")
        
        print(f"\n{'='*60}")
        print(f"📋 SAMPLE {sample_idx + 1}/{min(max_samples, len(cr))}")
        print(f"   Context: {len(context)} chars")
        print(f"   Questions: {len(questions)}")
        print(f"{'='*60}")
        
        # === SPLIT CONTEXT INTO FACTS ===
        documents = split_sf_context(context)
        print(f"\n📄 Split context into {len(documents)} facts")
        
        # === BATCH INGEST WITH HIGH CONCURRENCY ===
        print(f"🧠 Batch ingesting with {min(1000, len(documents))} workers...")
        start_ingest = time.time()
        
        # Use ingest_batch for high-performance parallel ingestion
        result = memory.ingest_batch(
            documents=documents,
            max_workers=min(1000, len(documents)),
            show_progress=True,
        )
        
        ingest_time = time.time() - start_ingest
        print(f"   ✅ Ingested {result.get('documents_processed', len(documents))} facts in {ingest_time:.1f}s")
        if result.get('documents_failed', 0) > 0:
            print(f"   ⚠️ {result['documents_failed']} documents failed")
        
        # === EVOLVE ===
        if run_evolution:
            print("\n🔄 Running evolution (decay should prioritize newer facts)...")
            start_evolve = time.time()
            memory.evolve()
            evolve_time = time.time() - start_evolve
            print(f"   ✅ Evolved in {evolve_time:.1f}s")
        
        # === QUERY ===
        print(f"\n🔍 Evaluating {min(max_questions_per_sample, len(questions))} questions...")
        
        sample_results = []
        
        def evaluate_single(q: str, a: Any) -> QueryResult:
            start = time.time()
            try:
                response = memory.query(q)
                predicted = response.answer
            except Exception as e:
                predicted = f"Error: {e}"
            latency = (time.time() - start) * 1000
            
            # Handle list answers
            if isinstance(a, list):
                a_str = a[0] if a else ""
            else:
                a_str = str(a)
            
            metrics = calculate_metrics(predicted, a)
            
            return QueryResult(
                query=q,
                expected=a_str,
                predicted=predicted,
                latency_ms=latency,
                **metrics
            )
        
        # Run queries concurrently
        test_qs = questions[:max_questions_per_sample]
        test_as = answers[:max_questions_per_sample]
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(evaluate_single, q, a): (i, q, a)
                for i, (q, a) in enumerate(zip(test_qs, test_as))
            }
            
            for future in as_completed(futures):
                idx, q, a = futures[future]
                try:
                    result = future.result()
                    sample_results.append(result)
                    
                    if show_details:
                        status = "✅" if result.substring_match > 0 else "❌"
                        print(f"\n{status} Q{idx+1}: {q[:70]}...")
                        print(f"   Expected: {result.expected[:60]}...")
                        print(f"   Got:      {result.predicted[:60]}...")
                        print(f"   SubStr: {result.substring_match:.0%} | F1: {result.f1:.0%}")
                        
                except Exception as e:
                    print(f"❌ Q{idx+1} failed: {e}")
        
        all_results.extend(sample_results)
        
        # Sample summary
        sample_substr = np.mean([r.substring_match for r in sample_results]) * 100
        sample_f1 = np.mean([r.f1 for r in sample_results]) * 100
        print(f"\n📊 Sample {sample_idx + 1} Results:")
        print(f"   Substring Match: {sample_substr:.1f}%")
        print(f"   F1 Score:        {sample_f1:.1f}%")
    
    # === AGGREGATE METRICS ===
    n = len(all_results)
    f1_scores = [r.f1 for r in all_results]
    substr_scores = [r.substring_match for r in all_results]
    latencies = [r.latency_ms for r in all_results]
    
    metrics = {
        "total_questions": n,
        "f1_score": np.mean(f1_scores) * 100,
        "substring_match": np.mean(substr_scores) * 100,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
    }
    
    return metrics, all_results


def print_conflict_resolution_results(metrics: Dict[str, float]):
    """Print Conflict Resolution results in Table 2 format."""
    
    print("\n" + "=" * 70)
    print("📊 CONFLICT RESOLUTION (SF) RESULTS")
    print("=" * 70)
    
    # From paper Table 2 - SF column (Conflict Resolution)
    # FC-SH = FactConsolidation Single Hop
    # FC-MH = FactConsolidation Multi Hop
    
    fc_score = metrics["substring_match"]
    
    print(f"\n┌{'─'*68}┐")
    print(f"│{'Agent Type':<25}│{'FC-SH':^12}│{'FC-MH':^12}│{'SF Avg':^15}│")
    print(f"├{'─'*68}┤")
    
    references = [
        ("GPT-4o-mini", 45.0, 5.0, 25.0),
        ("GPT-4.1-mini", 36.0, 5.0, 20.5),
        ("Text-Embed-3-Small", 28.0, 3.0, 15.5),
        ("GraphRAG", 14.0, 2.0, 8.0),
        ("HippoRAG-v2", 54.0, 5.0, 29.5),
        ("Mem0", 18.0, 2.0, 10.0),
    ]
    
    for name, fc_sh, fc_mh, avg in references:
        print(f"│{name:<25}│{fc_sh:^12.1f}│{fc_mh:^12.1f}│{avg:^15.1f}│")
    
    print(f"├{'─'*68}┤")
    print(f"│{'🧠 GraphMem (ours)':<25}│{fc_score:^12.1f}│{fc_score:^12.1f}│{fc_score:^15.1f}│")
    print(f"└{'─'*68}┘")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   • Substring Match: {metrics['substring_match']:.1f}%")
    print(f"   • F1 Score:        {metrics['f1_score']:.1f}%")
    print(f"   • Avg Latency:     {metrics['avg_latency_ms']:.0f}ms")
    
    # Comparison
    if fc_score > 29.5:
        print("\n🏆 GraphMem BEATS HippoRAG-v2 on Conflict Resolution!")
    elif fc_score > 25.0:
        print("\n✅ GraphMem beats GPT-4o-mini on Conflict Resolution!")
    elif fc_score > 20.5:
        print("\n✅ GraphMem beats GPT-4.1-mini on Conflict Resolution!")
    else:
        print("\n⚠️ Room for improvement on Conflict Resolution")


# =============================================================================
# MAIN - Run if executed directly
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║          CONFLICT RESOLUTION EVALUATION FOR GRAPHMEM               ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  This tests GraphMem's ability to prefer NEWER facts over OLDER    ║
    ║  ones when there are conflicts. The decay mechanism should help!   ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    To run in a notebook:
    
    from graphmem import GraphMem, MemoryConfig
    from conflict_resolution_eval import evaluate_conflict_resolution, print_conflict_resolution_results
    
    # Create fresh GraphMem instance
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
    memory = GraphMem(config)
    
    # Run evaluation
    metrics, results = evaluate_conflict_resolution(
        memory,
        max_samples=2,           # Test 2 samples
        max_questions_per_sample=20,  # 20 questions each
        run_evolution=True,      # Run evolve() - CRITICAL for conflict resolution!
    )
    
    # Print results
    print_conflict_resolution_results(metrics)
    """)


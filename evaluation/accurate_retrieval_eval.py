# =============================================================================
# 📊 GraphMem - ACCURATE RETRIEVAL (AR) Evaluation
# 
# Tests GraphMem's ability to accurately retrieve information from memory.
# Includes: SH-QA (Single-Hop), MH-QA (Multi-Hop), LME (LongMemEval), EventQA
# =============================================================================

import time
import string
import re
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from datasets import load_dataset

# =============================================================================
# CONTEXT SPLITTING - AR uses "Document 1:", "Document 2:", etc.
# =============================================================================

def split_ar_context(text: str) -> List[Dict[str, str]]:
    """
    Split AR (Accurate Retrieval) context into individual documents.
    Format: "Document 1:\nContent...\n\nDocument 2:\nContent..."
    """
    # Split on: "Document" + number + colon + optional spaces
    parts = re.split(r'Document \d+:\s*', text)
    # First item before "Document 1:" is usually empty, skip it
    parts = [p.strip() for p in parts if p.strip()]
    
    # Convert to {"content": "..."} format
    documents = []
    for i, part in enumerate(parts):
        if part:
            documents.append({
                "content": part,
                "metadata": {"document_id": i + 1}
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

def exact_match(prediction: str, ground_truth: str) -> float:
    """Check exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

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
        "exact_match": max(exact_match(prediction, gt) for gt in gt_list),
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
    task_type: str = ""  # sh_qa, mh_qa, lme, eventqa
    exact_match: float = 0.0
    f1: float = 0.0
    substring_match: float = 0.0
    latency_ms: float = 0.0

@dataclass
class ARResults:
    """Results broken down by task type."""
    sh_qa: List[QueryResult] = field(default_factory=list)
    mh_qa: List[QueryResult] = field(default_factory=list)
    lme: List[QueryResult] = field(default_factory=list)
    eventqa: List[QueryResult] = field(default_factory=list)
    
    def get_task_accuracy(self, task: str) -> float:
        results = getattr(self, task, [])
        if not results:
            return 0.0
        return np.mean([r.substring_match for r in results]) * 100

# =============================================================================
# ACCURATE RETRIEVAL EVALUATION
# =============================================================================

def evaluate_accurate_retrieval(
    config,  # MemoryConfig for creating fresh instances
    max_samples: int = 10,
    max_questions_per_sample: int = 10,
    max_concurrent: int = 5,
    show_details: bool = True,
    run_evolution: bool = True,
    turso_db_prefix: str = "ar_eval",
) -> Tuple[Dict[str, Any], ARResults]:
    """
    Evaluate GraphMem on Accurate Retrieval task.
    
    Tests:
    - SH-QA: Single-hop question answering
    - MH-QA: Multi-hop question answering  
    - LME: LongMemEval style queries
    - EventQA: Event-based questions
    
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
        Tuple of (metrics dict, ARResults object)
    """
    from graphmem import GraphMem
    import os
    import json
    
    print("📥 Loading Accurate_Retrieval dataset...")
    ds = load_dataset('ai-hyz/MemoryAgentBench')
    ar = ds['Accurate_Retrieval']
    print(f"   Found {len(ar)} samples\n")
    
    ar_results = ARResults()
    
    for sample_idx in range(min(max_samples, len(ar))):
        sample = ar[sample_idx]
        context = sample.get('context', '')
        questions = sample.get('questions', [])
        answers = sample.get('answers', [])
        metadata = sample.get('metadata', {})
        
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
        
        # Parse metadata to get source/task type
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        source = metadata.get('source', 'unknown')
        
        # Determine task type from source
        if 'ruler_qa1' in source:
            task_type = 'sh_qa'
        elif 'ruler_qa2' in source:
            task_type = 'mh_qa'
        elif 'longmemeval' in source:
            task_type = 'lme'
        elif 'eventqa' in source:
            task_type = 'eventqa'
        else:
            task_type = 'sh_qa'  # Default
        
        print(f"\n{'='*60}")
        print(f"📋 SAMPLE {sample_idx + 1}/{min(max_samples, len(ar))} [{task_type.upper()}]")
        print(f"   Source: {source}")
        print(f"   Context: {len(context)} chars")
        print(f"   Questions: {len(questions)}")
        print(f"{'='*60}")
        
        # === SPLIT CONTEXT INTO DOCUMENTS ===
        documents = split_ar_context(context)
        print(f"\n📄 Split context into {len(documents)} documents")
        
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
        print(f"   ✅ Ingested {result.get('documents_processed', len(documents))} docs in {ingest_time:.1f}s")
        if result.get('documents_failed', 0) > 0:
            print(f"   ⚠️ {result['documents_failed']} documents failed")
        
        # === EVOLVE ===
        if run_evolution:
            print("\n🔄 Running evolution...")
            start_evolve = time.time()
            memory.evolve()
            evolve_time = time.time() - start_evolve
            print(f"   ✅ Evolved in {evolve_time:.1f}s")
        
        # === QUERY ===
        print(f"\n🔍 Evaluating {min(max_questions_per_sample, len(questions))} questions...")
        
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
                task_type=task_type,
                latency_ms=latency,
                **metrics
            )
        
        # Run queries concurrently
        test_qs = questions[:max_questions_per_sample]
        test_as = answers[:max_questions_per_sample]
        
        sample_results = []
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
                    
                    # Add to appropriate task list
                    getattr(ar_results, task_type).append(result)
                    
                    if show_details:
                        status = "✅" if result.substring_match > 0 else "❌"
                        print(f"\n{status} Q{idx+1}: {q[:60]}...")
                        print(f"   Expected: {result.expected[:50]}...")
                        print(f"   Got:      {result.predicted[:50]}...")
                        print(f"   SubStr: {result.substring_match:.0%} | F1: {result.f1:.0%}")
                        
                except Exception as e:
                    print(f"❌ Q{idx+1} failed: {e}")
        
        # Sample summary
        if sample_results:
            sample_substr = np.mean([r.substring_match for r in sample_results]) * 100
            print(f"\n📊 Sample {sample_idx + 1} Results: {sample_substr:.1f}%")
    
    # === AGGREGATE METRICS ===
    all_results = ar_results.sh_qa + ar_results.mh_qa + ar_results.lme + ar_results.eventqa
    
    if not all_results:
        return {"error": "No results collected"}, ar_results
    
    metrics = {
        "total_questions": len(all_results),
        # Per-task scores
        "sh_qa": ar_results.get_task_accuracy('sh_qa'),
        "mh_qa": ar_results.get_task_accuracy('mh_qa'),
        "lme": ar_results.get_task_accuracy('lme'),
        "eventqa": ar_results.get_task_accuracy('eventqa'),
        # Overall
        "substring_match": np.mean([r.substring_match for r in all_results]) * 100,
        "f1_score": np.mean([r.f1 for r in all_results]) * 100,
        "exact_match": np.mean([r.exact_match for r in all_results]) * 100,
        # Latency
        "avg_latency_ms": np.mean([r.latency_ms for r in all_results]),
        "p50_latency_ms": np.percentile([r.latency_ms for r in all_results], 50),
    }
    
    # Calculate average across tasks that have data
    task_scores = [
        metrics['sh_qa'] if ar_results.sh_qa else None,
        metrics['mh_qa'] if ar_results.mh_qa else None,
        metrics['lme'] if ar_results.lme else None,
        metrics['eventqa'] if ar_results.eventqa else None,
    ]
    valid_scores = [s for s in task_scores if s is not None]
    metrics['ar_avg'] = np.mean(valid_scores) if valid_scores else 0
    
    return metrics, ar_results


def print_accurate_retrieval_results(metrics: Dict[str, float]):
    """Print Accurate Retrieval results in Table 2 format."""
    
    print("\n" + "=" * 80)
    print("📊 ACCURATE RETRIEVAL (AR) RESULTS")
    print("=" * 80)
    
    print(f"\n┌{'─'*78}┐")
    print(f"│{'Agent Type':<20}│{'SH-QA':^12}│{'MH-QA':^12}│{'LME(S*)':^12}│{'EventQA':^12}│{'Avg':^10}│")
    print(f"├{'─'*78}┤")
    
    references = [
        ("GPT-4o-mini", 64.0, 43.0, 30.7, 59.0, 49.2),
        ("GPT-4.1-mini", 83.0, 66.0, 55.7, 82.6, 71.8),
        ("Text-Embed-3-Small", 60.0, 44.0, 48.3, 63.0, 53.8),
        ("GraphRAG", 47.0, 47.0, 35.0, 34.4, 40.9),
        ("HippoRAG-v2", 76.0, 66.0, 50.7, 67.6, 65.1),
        ("Mem0", 25.0, 32.0, 36.0, 37.5, 32.6),
    ]
    
    for name, sh, mh, lme, event, avg in references:
        print(f"│{name:<20}│{sh:^12.1f}│{mh:^12.1f}│{lme:^12.1f}│{event:^12.1f}│{avg:^10.1f}│")
    
    print(f"├{'─'*78}┤")
    
    # GraphMem scores
    sh = metrics.get('sh_qa', metrics['substring_match'])
    mh = metrics.get('mh_qa', metrics['substring_match'])
    lme = metrics.get('lme', metrics['substring_match'])
    event = metrics.get('eventqa', metrics['substring_match'])
    avg = metrics.get('ar_avg', metrics['substring_match'])
    
    print(f"│{'🧠 GraphMem (ours)':<20}│{sh:^12.1f}│{mh:^12.1f}│{lme:^12.1f}│{event:^12.1f}│{avg:^10.1f}│")
    print(f"└{'─'*78}┘")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   • Total Questions: {metrics['total_questions']}")
    print(f"   • Substring Match: {metrics['substring_match']:.1f}%")
    print(f"   • F1 Score:        {metrics['f1_score']:.1f}%")
    print(f"   • Exact Match:     {metrics['exact_match']:.1f}%")
    print(f"   • Avg Latency:     {metrics['avg_latency_ms']:.0f}ms")
    
    # Comparison
    if avg > 71.8:
        print("\n🏆 GraphMem BEATS GPT-4.1-mini (best long-context)!")
    elif avg > 65.1:
        print("\n🏆 GraphMem BEATS HippoRAG-v2 (best structure-augmented)!")
    elif avg > 53.8:
        print("\n✅ GraphMem beats Text-Embed-3-Small RAG!")
    elif avg > 49.2:
        print("\n✅ GraphMem beats GPT-4o-mini baseline!")
    else:
        print("\n⚠️ Room for improvement on Accurate Retrieval")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║          ACCURATE RETRIEVAL (AR) EVALUATION FOR GRAPHMEM           ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Tests retrieval accuracy across:                                   ║
    ║  • SH-QA: Single-hop question answering                            ║
    ║  • MH-QA: Multi-hop question answering                             ║
    ║  • LME: LongMemEval style queries                                  ║
    ║  • EventQA: Event-based questions                                  ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    To run in a notebook:
    
    from graphmem import GraphMem, MemoryConfig
    from accurate_retrieval_eval import evaluate_accurate_retrieval, print_accurate_retrieval_results
    
    config = MemoryConfig(...)
    memory = GraphMem(config)
    
    metrics, results = evaluate_accurate_retrieval(
        memory,
        max_samples=10,
        max_questions_per_sample=10,
        run_evolution=True,
    )
    
    print_accurate_retrieval_results(metrics)
    """)


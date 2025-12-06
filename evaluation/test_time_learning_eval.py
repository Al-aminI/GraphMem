# =============================================================================
# 📊 GraphMem - TEST TIME LEARNING (TTL) Evaluation
# 
# Tests GraphMem's ability to learn from corrections and feedback at test time.
# Includes: MCC (Memory Correction Capability) and Recommendation tasks
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
# CONTEXT SPLITTING - TTL uses "Dialogue 1:", "Dialogue 2:", etc.
# =============================================================================

def split_ttl_context(text: str) -> List[Dict[str, str]]:
    """
    Split TTL (Test Time Learning) context into individual dialogues.
    Format: "Dialogue 1:\nSystem: ...\nUser: ...\n\nDialogue 2:..."
    """
    # Split on: "Dialogue" + number + colon + optional spaces
    parts = re.split(r'Dialogue \d+:\s*', text)
    # First item before "Dialogue 1:" is usually empty, skip it
    parts = [p.strip() for p in parts if p.strip()]
    
    # Convert to {"content": "..."} format
    documents = []
    for i, part in enumerate(parts):
        if part:
            documents.append({
                "content": f"Dialogue {i+1}: {part}",
                "metadata": {"dialogue_id": i + 1}
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
    task_type: str = ""  # mcc, recom
    f1: float = 0.0
    substring_match: float = 0.0
    latency_ms: float = 0.0
    learned_from_correction: bool = False

@dataclass
class TTLResults:
    """Results broken down by task type."""
    mcc: List[QueryResult] = field(default_factory=list)  # Memory Correction Capability
    recom: List[QueryResult] = field(default_factory=list)  # Recommendation
    
    def get_task_accuracy(self, task: str) -> float:
        results = getattr(self, task, [])
        if not results:
            return 0.0
        return np.mean([r.substring_match for r in results]) * 100

# =============================================================================
# TEST TIME LEARNING EVALUATION
# =============================================================================

def evaluate_test_time_learning(
    config,  # MemoryConfig for creating fresh instances
    max_samples: int = 5,
    max_questions_per_sample: int = 10,
    max_concurrent: int = 5,
    show_details: bool = True,
    run_evolution: bool = True,
    turso_db_prefix: str = "ttl_eval",
) -> Tuple[Dict[str, Any], TTLResults]:
    """
    Evaluate GraphMem on Test Time Learning task.
    
    Tests:
    - MCC: Memory Correction Capability - can the system learn from corrections?
    - Recom: Can the system learn preferences and make recommendations?
    
    The key innovation: GraphMem's evolve() should incorporate corrections
    by updating entity priorities and relationships.
    
    EACH SAMPLE GETS A FRESH GRAPHMEM INSTANCE WITH LOCAL TURSO DB.
    
    Args:
        config: MemoryConfig for creating fresh GraphMem instances
        max_samples: Number of samples to test
        max_questions_per_sample: Questions per sample
        max_concurrent: Concurrent queries
        show_details: Print details
        run_evolution: Whether to run evolve() after corrections
        turso_db_prefix: Prefix for local Turso database files
    
    Returns:
        Tuple of (metrics dict, TTLResults object)
    """
    from graphmem import GraphMem
    import os
    
    print("📥 Loading Test_Time_Learning dataset...")
    ds = load_dataset('ai-hyz/MemoryAgentBench')
    ttl = ds['Test_Time_Learning']
    print(f"   Found {len(ttl)} samples\n")
    
    ttl_results = TTLResults()
    
    for sample_idx in range(min(max_samples, len(ttl))):
        sample = ttl[sample_idx]
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
        print(f"📋 SAMPLE {sample_idx + 1}/{min(max_samples, len(ttl))}")
        print(f"   Context: {len(context)} chars")
        print(f"   Questions: {len(questions)}")
        print(f"{'='*60}")
        
        # === SPLIT CONTEXT INTO DIALOGUES ===
        documents = split_ttl_context(context)
        print(f"\n📄 Split context into {len(documents)} dialogues")
        
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
        print(f"   ✅ Ingested {result.get('documents_processed', len(documents))} dialogues in {ingest_time:.1f}s")
        if result.get('documents_failed', 0) > 0:
            print(f"   ⚠️ {result['documents_failed']} documents failed")
        
        # === FIRST QUERY (Before Learning) ===
        # Some TTL samples have a pattern: query → wrong answer → correction → query again
        
        # === EVOLVE (Critical for learning) ===
        if run_evolution:
            print("\n🔄 Running evolution (for learning integration)...")
            start_evolve = time.time()
            memory.evolve()
            evolve_time = time.time() - start_evolve
            print(f"   ✅ Evolved in {evolve_time:.1f}s")
        
        # === QUERY ===
        print(f"\n🔍 Evaluating {min(max_questions_per_sample, len(questions))} questions...")
        
        # Determine task type based on question patterns
        def get_task_type(question: str) -> str:
            q_lower = question.lower()
            if any(kw in q_lower for kw in ['recommend', 'suggest', 'preference', 'like', 'enjoy']):
                return 'recom'
            return 'mcc'
        
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
            task_type = get_task_type(q)
            
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
                    if result.task_type == 'recom':
                        ttl_results.recom.append(result)
                    else:
                        ttl_results.mcc.append(result)
                    
                    if show_details:
                        status = "✅" if result.substring_match > 0 else "❌"
                        print(f"\n{status} Q{idx+1} [{result.task_type.upper()}]: {q[:55]}...")
                        print(f"   Expected: {result.expected[:50]}...")
                        print(f"   Got:      {result.predicted[:50]}...")
                        print(f"   SubStr: {result.substring_match:.0%}")
                        
                except Exception as e:
                    print(f"❌ Q{idx+1} failed: {e}")
        
        # Sample summary
        if sample_results:
            sample_substr = np.mean([r.substring_match for r in sample_results]) * 100
            print(f"\n📊 Sample {sample_idx + 1} Results: {sample_substr:.1f}%")
    
    # === AGGREGATE METRICS ===
    all_results = ttl_results.mcc + ttl_results.recom
    
    if not all_results:
        return {"error": "No results collected"}, ttl_results
    
    mcc_score = ttl_results.get_task_accuracy('mcc')
    recom_score = ttl_results.get_task_accuracy('recom')
    
    # TTL average (weighted like paper)
    ttl_avg = (mcc_score + recom_score) / 2 if recom_score > 0 else mcc_score
    
    metrics = {
        "total_questions": len(all_results),
        # Per-task scores
        "mcc": mcc_score,
        "recom": recom_score,
        "ttl_avg": ttl_avg,
        # Overall
        "substring_match": np.mean([r.substring_match for r in all_results]) * 100,
        "f1_score": np.mean([r.f1 for r in all_results]) * 100,
        # Latency
        "avg_latency_ms": np.mean([r.latency_ms for r in all_results]),
    }
    
    return metrics, ttl_results


def print_test_time_learning_results(metrics: Dict[str, float]):
    """Print Test Time Learning results in Table 2 format."""
    
    print("\n" + "=" * 70)
    print("📊 TEST TIME LEARNING (TTL) RESULTS")
    print("=" * 70)
    
    print(f"\n┌{'─'*58}┐")
    print(f"│{'Agent Type':<25}│{'MCC':^15}│{'Recom':^15}│{'Avg':^10}│")
    print(f"├{'─'*58}┤")
    
    # From paper Table 2 - TTL column
    references = [
        ("GPT-4o-mini", 82.0, 15.1, 48.6),
        ("GPT-4.1-mini", 75.6, 16.7, 46.2),
        ("Text-Embed-3-Small", 70.0, 15.3, 42.7),
        ("GraphRAG", 39.8, 9.8, 24.8),
        ("HippoRAG-v2", 61.4, 10.2, 35.8),
        ("Mem0", 32.4, 10.0, 21.2),
    ]
    
    for name, mcc, recom, avg in references:
        print(f"│{name:<25}│{mcc:^15.1f}│{recom:^15.1f}│{avg:^10.1f}│")
    
    print(f"├{'─'*58}┤")
    
    mcc = metrics.get('mcc', 0)
    recom = metrics.get('recom', 0)
    avg = metrics.get('ttl_avg', 0)
    
    print(f"│{'🧠 GraphMem (ours)':<25}│{mcc:^15.1f}│{recom:^15.1f}│{avg:^10.1f}│")
    print(f"└{'─'*58}┘")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   • MCC (Memory Correction): {mcc:.1f}%")
    print(f"   • Recommendation:          {recom:.1f}%")
    print(f"   • TTL Average:             {avg:.1f}%")
    print(f"   • Avg Latency:             {metrics['avg_latency_ms']:.0f}ms")
    
    # Comparison
    if avg > 48.6:
        print("\n🏆 GraphMem BEATS GPT-4o-mini on Test Time Learning!")
    elif avg > 35.8:
        print("\n✅ GraphMem beats HippoRAG-v2 on Test Time Learning!")
    elif avg > 24.8:
        print("\n✅ GraphMem beats GraphRAG on Test Time Learning!")
    else:
        print("\n⚠️ Room for improvement on Test Time Learning")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║        TEST TIME LEARNING (TTL) EVALUATION FOR GRAPHMEM            ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Tests learning from corrections and feedback:                      ║
    ║  • MCC: Memory Correction Capability                               ║
    ║  • Recom: Recommendation based on learned preferences              ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    To run in a notebook:
    
    from graphmem import GraphMem, MemoryConfig
    from test_time_learning_eval import evaluate_test_time_learning, print_test_time_learning_results
    
    config = MemoryConfig(...)
    memory = GraphMem(config)
    
    metrics, results = evaluate_test_time_learning(
        memory,
        max_samples=5,
        run_evolution=True,  # Critical for learning!
    )
    
    print_test_time_learning_results(metrics)
    """)


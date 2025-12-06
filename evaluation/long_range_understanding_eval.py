# =============================================================================
# 📊 GraphMem - LONG RANGE UNDERSTANDING (LRU) Evaluation
# 
# Tests GraphMem's ability to understand and synthesize information
# across long contexts.
# Includes: Summarization and Detection QA tasks
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
# CONTEXT SPLITTING - LRU uses novels/long text, split by chapters or paragraphs
# =============================================================================

def split_lru_context(text: str, max_chunk_size: int = 5000) -> List[Dict[str, str]]:
    """
    Split LRU (Long Range Understanding) context into manageable chunks.
    
    Strategy:
    1. Try to split by chapters first ("Chapter 1", "Chapter 2", etc.)
    2. If no chapters, split by paragraphs (double newlines)
    3. If paragraphs are too large, split by fixed size
    """
    documents = []
    
    # Strategy 1: Try chapter splits
    chapter_pattern = r'(?:Chapter|CHAPTER|Ch\.?)\s*\d+'
    chapter_splits = re.split(f'({chapter_pattern})', text)
    
    if len(chapter_splits) > 3:  # Found chapters
        # Combine chapter headers with their content
        current_chunk = ""
        for i, part in enumerate(chapter_splits):
            if re.match(chapter_pattern, part):
                if current_chunk.strip():
                    documents.append({"content": current_chunk.strip()})
                current_chunk = part
            else:
                current_chunk += " " + part
        if current_chunk.strip():
            documents.append({"content": current_chunk.strip()})
    
    # Strategy 2: Split by paragraphs if no chapters or too few
    if len(documents) < 5:
        documents = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size, save current and start new
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                documents.append({"content": current_chunk.strip()})
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk.strip():
            documents.append({"content": current_chunk.strip()})
    
    # Strategy 3: If still too few chunks, split by fixed size
    if len(documents) < 10 and len(text) > max_chunk_size * 10:
        documents = []
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i + max_chunk_size].strip()
            if chunk:
                documents.append({"content": chunk})
    
    # Add metadata
    for i, doc in enumerate(documents):
        doc["metadata"] = {"chunk_id": i + 1}
    
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

def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Calculate ROUGE-L score (longest common subsequence)."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == truth_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return (2 * precision * recall) / (precision + recall)

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
        "rouge_l": max(rouge_l_score(prediction, gt) for gt in gt_list),
    }

# =============================================================================
# RESULT STORAGE
# =============================================================================

@dataclass
class QueryResult:
    query: str
    expected: str
    predicted: str
    task_type: str = ""  # summ, detqa
    f1: float = 0.0
    substring_match: float = 0.0
    rouge_l: float = 0.0
    latency_ms: float = 0.0

@dataclass
class LRUResults:
    """Results broken down by task type."""
    summ: List[QueryResult] = field(default_factory=list)  # Summarization
    detqa: List[QueryResult] = field(default_factory=list)  # Detection QA
    
    def get_task_accuracy(self, task: str) -> float:
        results = getattr(self, task, [])
        if not results:
            return 0.0
        return np.mean([r.substring_match for r in results]) * 100

# =============================================================================
# LONG RANGE UNDERSTANDING EVALUATION
# =============================================================================

def evaluate_long_range_understanding(
    config,  # MemoryConfig for creating fresh instances
    max_samples: int = 5,
    max_questions_per_sample: int = 5,
    max_concurrent: int = 3,
    show_details: bool = True,
    run_evolution: bool = True,
    turso_db_prefix: str = "lru_eval",
) -> Tuple[Dict[str, Any], LRUResults]:
    """
    Evaluate GraphMem on Long Range Understanding task.
    
    Tests:
    - Summ: Summarization - synthesize long contexts
    - DetQA: Detection QA - find specific details in long contexts
    
    GraphMem's community detection and hierarchical summarization
    should help with understanding long-range dependencies.
    
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
        Tuple of (metrics dict, LRUResults object)
    """
    from graphmem import GraphMem
    import os
    
    print("📥 Loading Long_Range_Understanding dataset...")
    ds = load_dataset('ai-hyz/MemoryAgentBench')
    lru = ds['Long_Range_Understanding']
    print(f"   Found {len(lru)} samples\n")
    
    lru_results = LRUResults()
    
    for sample_idx in range(min(max_samples, len(lru))):
        sample = lru[sample_idx]
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
        print(f"📋 SAMPLE {sample_idx + 1}/{min(max_samples, len(lru))}")
        print(f"   Context: {len(context)} chars ({len(context)//1000}K tokens approx)")
        print(f"   Questions: {len(questions)}")
        print(f"{'='*60}")
        
        # === SPLIT CONTEXT INTO CHUNKS ===
        documents = split_lru_context(context, max_chunk_size=5000)
        print(f"\n📄 Split context into {len(documents)} chunks")
        
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
        print(f"   ✅ Ingested {result.get('documents_processed', len(documents))} chunks in {ingest_time:.1f}s")
        if result.get('documents_failed', 0) > 0:
            print(f"   ⚠️ {result['documents_failed']} documents failed")
        
        # === EVOLVE (Critical for summarization!) ===
        if run_evolution:
            print("\n🔄 Running evolution (builds community summaries)...")
            start_evolve = time.time()
            memory.evolve()
            evolve_time = time.time() - start_evolve
            print(f"   ✅ Evolved in {evolve_time:.1f}s")
        
        # === QUERY ===
        print(f"\n🔍 Evaluating {min(max_questions_per_sample, len(questions))} questions...")
        
        # Determine task type based on question patterns
        def get_task_type(question: str) -> str:
            q_lower = question.lower()
            if any(kw in q_lower for kw in ['summarize', 'summary', 'main point', 'overview', 'key theme']):
                return 'summ'
            return 'detqa'
        
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
        
        # Run queries (less concurrent for long contexts)
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
                    if result.task_type == 'summ':
                        lru_results.summ.append(result)
                    else:
                        lru_results.detqa.append(result)
                    
                    if show_details:
                        status = "✅" if result.substring_match > 0 else "❌"
                        print(f"\n{status} Q{idx+1} [{result.task_type.upper()}]: {q[:55]}...")
                        print(f"   Expected: {result.expected[:50]}...")
                        print(f"   Got:      {result.predicted[:50]}...")
                        print(f"   SubStr: {result.substring_match:.0%} | ROUGE-L: {result.rouge_l:.0%}")
                        
                except Exception as e:
                    print(f"❌ Q{idx+1} failed: {e}")
        
        # Sample summary
        if sample_results:
            sample_substr = np.mean([r.substring_match for r in sample_results]) * 100
            print(f"\n📊 Sample {sample_idx + 1} Results: {sample_substr:.1f}%")
    
    # === AGGREGATE METRICS ===
    all_results = lru_results.summ + lru_results.detqa
    
    if not all_results:
        return {"error": "No results collected"}, lru_results
    
    summ_score = lru_results.get_task_accuracy('summ')
    detqa_score = lru_results.get_task_accuracy('detqa')
    
    # LRU average
    lru_avg = (summ_score + detqa_score) / 2 if summ_score > 0 and detqa_score > 0 else max(summ_score, detqa_score)
    
    metrics = {
        "total_questions": len(all_results),
        # Per-task scores
        "summ": summ_score,
        "detqa": detqa_score,
        "lru_avg": lru_avg,
        # Overall
        "substring_match": np.mean([r.substring_match for r in all_results]) * 100,
        "f1_score": np.mean([r.f1 for r in all_results]) * 100,
        "rouge_l": np.mean([r.rouge_l for r in all_results]) * 100,
        # Latency
        "avg_latency_ms": np.mean([r.latency_ms for r in all_results]),
    }
    
    return metrics, lru_results


def print_long_range_understanding_results(metrics: Dict[str, float]):
    """Print Long Range Understanding results in Table 2 format."""
    
    print("\n" + "=" * 70)
    print("📊 LONG RANGE UNDERSTANDING (LRU) RESULTS")
    print("=" * 70)
    
    print(f"\n┌{'─'*58}┐")
    print(f"│{'Agent Type':<25}│{'Summ':^15}│{'DetQA':^15}│{'Avg':^10}│")
    print(f"├{'─'*58}┤")
    
    # From paper Table 2 - LRU column
    references = [
        ("GPT-4o-mini", 28.9, 63.4, 46.2),
        ("GPT-4.1-mini", 41.9, 56.3, 49.1),
        ("Claude-3.7-Sonnet", 52.5, 71.8, 62.2),
        ("Text-Embed-3-Small", 17.7, 54.9, 36.3),
        ("GraphRAG", 0.4, 39.4, 19.9),
        ("HippoRAG-v2", 14.6, 57.7, 36.2),
        ("Mem0", 4.8, 36.6, 20.7),
    ]
    
    for name, summ, detqa, avg in references:
        print(f"│{name:<25}│{summ:^15.1f}│{detqa:^15.1f}│{avg:^10.1f}│")
    
    print(f"├{'─'*58}┤")
    
    summ = metrics.get('summ', 0)
    detqa = metrics.get('detqa', 0)
    avg = metrics.get('lru_avg', 0)
    
    print(f"│{'🧠 GraphMem (ours)':<25}│{summ:^15.1f}│{detqa:^15.1f}│{avg:^10.1f}│")
    print(f"└{'─'*58}┘")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   • Summarization:  {summ:.1f}%")
    print(f"   • Detection QA:   {detqa:.1f}%")
    print(f"   • ROUGE-L:        {metrics.get('rouge_l', 0):.1f}%")
    print(f"   • LRU Average:    {avg:.1f}%")
    print(f"   • Avg Latency:    {metrics['avg_latency_ms']:.0f}ms")
    
    # Comparison
    if avg > 62.2:
        print("\n🏆 GraphMem BEATS Claude-3.7-Sonnet on Long Range Understanding!")
    elif avg > 49.1:
        print("\n🏆 GraphMem BEATS GPT-4.1-mini on Long Range Understanding!")
    elif avg > 36.3:
        print("\n✅ GraphMem beats Text-Embed-3-Small on Long Range Understanding!")
    elif avg > 19.9:
        print("\n✅ GraphMem beats GraphRAG on Long Range Understanding!")
    else:
        print("\n⚠️ Room for improvement on Long Range Understanding")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║     LONG RANGE UNDERSTANDING (LRU) EVALUATION FOR GRAPHMEM         ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Tests understanding across long contexts:                          ║
    ║  • Summ: Summarization of long documents                           ║
    ║  • DetQA: Detection/finding specific details                       ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    To run in a notebook:
    
    from graphmem import GraphMem, MemoryConfig
    from long_range_understanding_eval import (
        evaluate_long_range_understanding, 
        print_long_range_understanding_results
    )
    
    config = MemoryConfig(...)
    memory = GraphMem(config)
    
    metrics, results = evaluate_long_range_understanding(
        memory,
        max_samples=3,
        run_evolution=True,  # Critical for community summaries!
    )
    
    print_long_range_understanding_results(metrics)
    """)


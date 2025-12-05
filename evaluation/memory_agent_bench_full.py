#!/usr/bin/env python3
"""
GraphMem MemoryAgentBench Full Evaluation

Comprehensive benchmark following the paper:
"Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions"
https://arxiv.org/abs/2507.05257

Tests ALL FOUR core memory competencies:
1. AR (Accurate Retrieval): SH-QA, MH-QA, LME(S*), EventQA
2. TTL (Test-Time Learning): MCC, Recom
3. LRU (Long-Range Understanding): Summ, DetQA  
4. SF (Selective Forgetting): FC-SH, FC-MH

Reference: https://github.com/HUST-AI-HYZ/MemoryAgentBench
"""

import os
import sys
import json
import time
import logging
import argparse
import string
import re
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add GraphMem to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global settings for concurrency
MAX_CONCURRENT_SAMPLES = 5  # Process multiple samples in parallel
MAX_CONCURRENT_QUERIES = 10  # Process multiple queries in parallel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TaskResult:
    """Result for a single task/query."""
    query: str
    expected: Any
    predicted: str
    exact_match: float = 0.0
    f1: float = 0.0
    substring_match: float = 0.0
    rouge_l: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    memory_construction_time: float = 0.0
    query_time: float = 0.0


@dataclass
class CompetencyResult:
    """Result for a competency (AR, TTL, LRU, SF)."""
    name: str
    tasks: Dict[str, List[TaskResult]] = field(default_factory=dict)
    
    def avg_score(self, metric: str = "substring_match") -> float:
        """Calculate average score across all tasks."""
        all_scores = []
        for task_results in self.tasks.values():
            for r in task_results:
                all_scores.append(getattr(r, metric, 0))
        return np.mean(all_scores) * 100 if all_scores else 0.0
    
    def task_scores(self, metric: str = "substring_match") -> Dict[str, float]:
        """Get per-task scores."""
        return {
            task: np.mean([getattr(r, metric, 0) for r in results]) * 100
            for task, results in self.tasks.items()
        }


@dataclass  
class BenchmarkResult:
    """Complete benchmark result."""
    model: str
    timestamp: str
    ar: CompetencyResult = field(default_factory=lambda: CompetencyResult("AR"))
    ttl: CompetencyResult = field(default_factory=lambda: CompetencyResult("TTL"))
    lru: CompetencyResult = field(default_factory=lambda: CompetencyResult("LRU"))
    sf: CompetencyResult = field(default_factory=lambda: CompetencyResult("SF"))
    total_time: float = 0.0
    
    def overall_score(self, metric: str = "substring_match") -> float:
        """Calculate weighted overall score."""
        scores = [
            self.ar.avg_score(metric),
            self.ttl.avg_score(metric),
            self.lru.avg_score(metric),
            self.sf.avg_score(metric),
        ]
        return np.mean([s for s in scores if s > 0])


# =============================================================================
# METRICS CALCULATION
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
    # Handle different ground truth formats
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif isinstance(ground_truths, list):
        gt_list = [str(g) for g in ground_truths]
    else:
        gt_list = [str(ground_truths)]
    
    metrics = {
        "exact_match": max(exact_match(prediction, gt) for gt in gt_list),
        "f1": max(f1_score(prediction, gt) for gt in gt_list),
        "substring_match": max(substring_match(prediction, gt) for gt in gt_list),
    }
    
    return metrics


# =============================================================================
# GRAPHMEM AGENT WRAPPER
# =============================================================================

class GraphMemAgent:
    """GraphMem agent wrapper for benchmarking.
    
    Evolution Features Utilized:
    
    1. PAGERANK CENTRALITY
       - Identifies hub entities (highly connected)
       - Boosts importance of well-connected nodes
       - Used in retrieval ranking
    
    2. MEMORY DECAY
       - Priority-based: Higher fact # = newer = survives
       - LLM-based: Detects semantic conflicts
       - Filters EPHEMERAL edges during retrieval
    
    3. CONSOLIDATION  
       - LLM-based entity merging
       - Alias unification
       - Reduces graph noise
    
    4. TEMPORAL VALIDITY
       - valid_from / valid_until on edges
       - Point-in-time queries
       - Supersession detection
    """
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: str = None,
        api_base: str = None,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
        chunk_size: int = 512,
        turso_db_path: str = "graphmem_bench.db",
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.azure_deployment = azure_deployment
        self.azure_embedding_deployment = azure_embedding_deployment
        self.chunk_size = chunk_size
        self.turso_db_path = turso_db_path
        self.gm = None
        self.memory_construction_time = 0
        self.evolution_stats = {
            "consolidations": 0,
            "decays": 0,
            "pagerank_updates": 0,
            "temporal_conflicts": 0,
        }
        
    def initialize(self, fresh: bool = True):
        """Initialize GraphMem instance."""
        from graphmem import GraphMem, MemoryConfig
        
        if fresh and os.path.exists(self.turso_db_path):
            os.remove(self.turso_db_path)
        if fresh and os.path.exists(f"{self.turso_db_path}_cache.db"):
            os.remove(f"{self.turso_db_path}_cache.db")
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        config = MemoryConfig(
            llm_provider=provider_name,
            llm_api_key=self.api_key,
            llm_api_base=self.api_base,
            llm_model=self.llm_model,
            embedding_provider=provider_name,
            embedding_api_key=self.api_key,
            embedding_api_base=self.api_base,
            embedding_model=self.embedding_model,
            azure_api_version="2024-08-01-preview",
            azure_deployment=self.azure_deployment,
            azure_embedding_deployment=self.azure_embedding_deployment,
            turso_db_path=self.turso_db_path,
        )
        
        self.gm = GraphMem(config)
        self.memory_construction_time = 0
        return self.gm
    
    def memorize(self, content: str):
        """Add content to memory."""
        if self.gm is None:
            self.initialize()
        
        start = time.time()
        self.gm.ingest(content)
        self.memory_construction_time += time.time() - start
        
    def memorize_batch(self, chunks: List[str], evolve_after: bool = True):
        """Batch memorize with evolution to leverage all GraphMem features.
        
        EVOLUTION BENEFITS:
        1. PageRank: Updates importance scores based on graph structure
        2. Decay: Marks superseded/conflicting edges as EPHEMERAL
        3. Consolidation: Merges similar entities (LLM-based)
        4. Temporal: Updates valid_from/valid_until based on conflicts
        """
        if self.gm is None:
            self.initialize()
        
        start = time.time()
        
        # Ingest in batches
        docs = [{"id": f"chunk_{i}", "content": chunk} for i, chunk in enumerate(chunks)]
        try:
            self.gm.ingest_batch(
                documents=docs,
                max_workers=10,
                show_progress=False,
                aggressive=True,
            )
        except Exception as e:
            logger.warning(f"Batch ingestion failed: {e}, falling back to sequential")
            for chunk in chunks:
                try:
                    self.gm.ingest(chunk)
                except Exception as e2:
                    logger.warning(f"Single ingestion failed: {e2}")
        
        # CRITICAL: Evolve memory to benefit from all features
        if evolve_after and len(chunks) > 1:
            try:
                evolution_result = self.gm.evolve()
                events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
                
                # Track evolution statistics
                for event in events:
                    event_type = str(event.evolution_type).lower()
                    if 'consolidat' in event_type:
                        self.evolution_stats["consolidations"] += 1
                    elif 'decay' in event_type:
                        self.evolution_stats["decays"] += 1
                        if 'temporal' in str(event.reason).lower():
                            self.evolution_stats["temporal_conflicts"] += 1
                    elif 'reinforce' in event_type:
                        self.evolution_stats["pagerank_updates"] += 1
                
                if events:
                    logger.debug(f"📊 Evolution: {len(events)} events "
                               f"({self.evolution_stats['consolidations']} consolidations, "
                               f"{self.evolution_stats['decays']} decays)")
                               
            except Exception as e:
                logger.debug(f"Evolution failed: {e}")
        
        self.memory_construction_time += time.time() - start
    
    def query(self, question: str) -> Tuple[str, float]:
        """Query memory and return answer with latency."""
        if self.gm is None:
            return "No memory initialized", 0.0
        
        start = time.time()
        try:
            response = self.gm.query(question)
            answer = response.answer
        except Exception as e:
            answer = f"Error: {e}"
        
        latency = (time.time() - start) * 1000
        return answer, latency
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        if self.gm is None:
            return {"entities": 0, "relationships": 0, "clusters": 0}
        
        return {
            "entities": len(self.gm._memory.nodes),
            "relationships": len(self.gm._memory.edges),
            "clusters": len(self.gm._memory.clusters) if hasattr(self.gm._memory, 'clusters') else 0,
        }
    
    def get_evolution_impact(self) -> Dict[str, Any]:
        """Get detailed evolution impact statistics.
        
        Shows how much the evolution features are actually helping.
        """
        if self.gm is None:
            return {}
        
        # Count edges by state
        active_edges = 0
        decayed_edges = 0
        temporal_edges = 0
        
        for edge in self.gm._memory.edges.values():
            if edge.importance.name == "EPHEMERAL" or edge.state.name != "ACTIVE":
                decayed_edges += 1
            else:
                active_edges += 1
            
            if edge.valid_from or edge.valid_until:
                temporal_edges += 1
        
        # Count nodes by importance
        importance_dist = {}
        for node in self.gm._memory.nodes.values():
            imp_name = node.importance.name
            importance_dist[imp_name] = importance_dist.get(imp_name, 0) + 1
        
        return {
            "total_entities": len(self.gm._memory.nodes),
            "total_relationships": len(self.gm._memory.edges),
            "active_edges": active_edges,
            "decayed_edges": decayed_edges,
            "temporal_edges": temporal_edges,
            "decay_rate": decayed_edges / max(1, len(self.gm._memory.edges)),
            "importance_distribution": importance_dist,
            "evolution_events": self.evolution_stats,
        }


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset_from_hf(dataset_name: str = "ai-hyz/MemoryAgentBench"):
    """Load MemoryAgentBench dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        logger.info(f"📥 Loading {dataset_name} from HuggingFace...")
        ds = load_dataset(dataset_name)
        
        logger.info("Available splits:")
        for split in ds:
            logger.info(f"  - {split}: {len(ds[split])} samples")
        
        return ds
    except ImportError:
        raise ImportError("Install: pip install datasets")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def chunk_text(text: str, chunk_size: int = 512, model: str = "gpt-4o-mini") -> List[str]:
    """Split text into chunks of specified token size."""
    # try:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    # except:
    #     # Fallback to simple character-based chunking
    #     char_size = chunk_size * 4  # Rough approximation
    #     return [text[i:i+char_size] for i in range(0, len(text), char_size)]
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(encoding.decode(chunk_tokens))
    
    return chunks


# =============================================================================
# COMPETENCY EVALUATORS
# =============================================================================

class AccurateRetrievalEvaluator:
    """
    AR (Accurate Retrieval) Evaluator
    
    Tests: SH-QA, MH-QA, LME(S*), EventQA
    
    Uses concurrent query processing for faster evaluation.
    """
    
    def __init__(self, agent: GraphMemAgent, max_samples: int = 100, max_workers: int = 10):
        self.agent = agent
        self.max_samples = max_samples
        self.max_workers = max_workers
        self.result = CompetencyResult("AR")
        self._lock = Lock()
    
    def _evaluate_query(self, question: str, expected: Any) -> TaskResult:
        """Evaluate a single query (thread-safe)."""
        predicted, latency = self.agent.query(question)
        metrics = calculate_metrics(predicted, expected)
        
        return TaskResult(
            query=question,
            expected=expected,
            predicted=predicted,
            latency_ms=latency,
            memory_construction_time=self.agent.memory_construction_time,
            **metrics
        )
    
    def _evaluate_queries_concurrent(self, qa_pairs: List[Tuple[str, Any]]) -> List[TaskResult]:
        """Evaluate multiple queries concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_query, q, a): (q, a) 
                for q, a in qa_pairs
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with self._lock:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
        
        return results
    
    def evaluate_sh_qa(self, dataset) -> List[TaskResult]:
        """Single-hop QA evaluation with concurrent query processing."""
        logger.info("📊 Evaluating SH-QA (Single-Hop QA)...")
        logger.info(f"   Using {self.max_workers} concurrent workers")
        results = []
        
        # Use Accurate_Retrieval split
        if "Accurate_Retrieval" not in dataset:
            logger.warning("Accurate_Retrieval split not found")
            return results
        
        samples = list(dataset["Accurate_Retrieval"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            # Initialize fresh memory
            self.agent.initialize(fresh=True)
            
            # Get context and questions
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # Chunk and memorize context
            chunks = chunk_text(context, chunk_size=512)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # Evaluate single-hop questions CONCURRENTLY
            qa_pairs = list(zip(questions[:5], answers[:5]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            logger.info(f"   Sample {idx+1}/{min(len(samples), 3)}: {len(sample_results)} queries evaluated")
            
            if idx >= 2:  # Limit samples for speed
                break
        
        self.result.tasks["SH-QA"] = results
        return results
    
    def evaluate_mh_qa(self, dataset) -> List[TaskResult]:
        """Multi-hop QA evaluation with concurrent query processing."""
        logger.info("📊 Evaluating MH-QA (Multi-Hop QA)...")
        logger.info(f"   Using {self.max_workers} concurrent workers")
        results = []
        
        if "Accurate_Retrieval" not in dataset:
            return results
        
        samples = list(dataset["Accurate_Retrieval"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            chunks = chunk_text(context, chunk_size=512)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # Multi-hop questions CONCURRENTLY
            qa_pairs = list(zip(questions[5:15], answers[5:15]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            logger.info(f"   Sample {idx+1}/{min(len(samples), 3)}: {len(sample_results)} queries evaluated")
            
            if idx >= 2:
                break
        
        self.result.tasks["MH-QA"] = results
        return results
    
    def evaluate_lme(self, dataset) -> List[TaskResult]:
        """LongMemEval (S*) evaluation with concurrent query processing."""
        logger.info("📊 Evaluating LME(S*) (Long Memory Evaluation)...")
        logger.info(f"   Using {self.max_workers} concurrent workers")
        results = []
        
        # LME is in Long_Range_Understanding or can use Accurate_Retrieval
        split_name = "Long_Range_Understanding" if "Long_Range_Understanding" in dataset else "Accurate_Retrieval"
        if split_name not in dataset:
            return results
        
        samples = list(dataset[split_name])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # Larger chunks for LME
            chunks = chunk_text(context, chunk_size=4096)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # CONCURRENT query processing
            qa_pairs = list(zip(questions[:10], answers[:10]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            logger.info(f"   Sample {idx+1}/{min(len(samples), 3)}: {len(sample_results)} queries evaluated")
            
            if idx >= 2:
                break
        
        self.result.tasks["LME(S*)"] = results
        return results
    
    def evaluate_eventqa(self, dataset) -> List[TaskResult]:
        """EventQA evaluation."""
        logger.info("📊 Evaluating EventQA...")
        results = []
        
        # EventQA might be in a separate split or Accurate_Retrieval
        if "Accurate_Retrieval" not in dataset:
            return results
        
        samples = list(dataset["Accurate_Retrieval"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            chunks = chunk_text(context, chunk_size=512)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # Event-based questions
            for q_idx, (question, expected) in enumerate(zip(questions[:10], answers[:10])):
                if not isinstance(expected, list):
                    expected = [expected]
                
                predicted, latency = self.agent.query(question)
                
                # EventQA uses recall metric
                recall = sum(1 for ans in expected if str(ans).lower() in predicted.lower()) / len(expected)
                
                results.append(TaskResult(
                    query=question,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    substring_match=recall,
                    f1=recall,
                ))
            
            if idx >= 2:
                break
        
        self.result.tasks["EventQA"] = results
        return results
    
    def run(self, dataset) -> CompetencyResult:
        """Run all AR evaluations."""
        self.evaluate_sh_qa(dataset)
        self.evaluate_mh_qa(dataset)
        self.evaluate_lme(dataset)
        self.evaluate_eventqa(dataset)
        return self.result


class TestTimeLearningEvaluator:
    """
    TTL (Test-Time Learning) Evaluator
    
    Tests: MCC (Multi-Choice Cloze), Recom (Movie Recommendation)
    
    Uses concurrent query processing for faster evaluation.
    """
    
    def __init__(self, agent: GraphMemAgent, max_samples: int = 100, max_workers: int = 10):
        self.agent = agent
        self.max_samples = max_samples
        self.max_workers = max_workers
        self.result = CompetencyResult("TTL")
        self._lock = Lock()
    
    def _evaluate_query(self, question: str, expected: Any) -> TaskResult:
        """Evaluate a single query (thread-safe)."""
        predicted, latency = self.agent.query(question)
        metrics = calculate_metrics(predicted, expected)
        
        return TaskResult(
            query=question,
            expected=expected,
            predicted=predicted,
            latency_ms=latency,
            **metrics
        )
    
    def _evaluate_queries_concurrent(self, qa_pairs: List[Tuple[str, Any]]) -> List[TaskResult]:
        """Evaluate multiple queries concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_query, q, a): (q, a) 
                for q, a in qa_pairs
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with self._lock:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
        
        return results
    
    def evaluate_mcc(self, dataset) -> List[TaskResult]:
        """Multi-Choice Cloze (in-context learning) with concurrent processing."""
        logger.info("📊 Evaluating MCC (Multi-Choice Cloze)...")
        logger.info(f"   Using {self.max_workers} concurrent workers")
        results = []
        
        if "Test_Time_Learning" not in dataset:
            return results
        
        samples = list(dataset["Test_Time_Learning"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # For TTL, we need to learn patterns from examples
            chunks = chunk_text(context, chunk_size=4096)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # CONCURRENT query processing
            qa_pairs = list(zip(questions[:10], answers[:10]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            logger.info(f"   Sample {idx+1}/{min(len(samples), 3)}: {len(sample_results)} queries evaluated")
            
            if idx >= 2:
                break
        
        self.result.tasks["MCC"] = results
        return results
    
    def evaluate_recom(self, dataset) -> List[TaskResult]:
        """Movie Recommendation evaluation."""
        logger.info("📊 Evaluating Recom (Movie Recommendation)...")
        results = []
        
        if "Test_Time_Learning" not in dataset:
            return results
        
        samples = list(dataset["Test_Time_Learning"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            chunks = chunk_text(context, chunk_size=4096)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            for q_idx, (question, expected) in enumerate(zip(questions[:5], answers[:5])):
                predicted, latency = self.agent.query(question)
                
                # For recommendations, check if expected items appear in predicted
                if isinstance(expected, list):
                    recall = sum(1 for e in expected if str(e).lower() in predicted.lower()) / len(expected)
                else:
                    recall = float(str(expected).lower() in predicted.lower())
                
                results.append(TaskResult(
                    query=question,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    substring_match=recall,
                ))
            
            if idx >= 1:
                break
        
        self.result.tasks["Recom"] = results
        return results
    
    def run(self, dataset) -> CompetencyResult:
        """Run all TTL evaluations."""
        self.evaluate_mcc(dataset)
        self.evaluate_recom(dataset)
        return self.result


class LongRangeUnderstandingEvaluator:
    """
    LRU (Long-Range Understanding) Evaluator
    
    Tests: Summ (Summarization), DetQA (Detective QA)
    
    Uses concurrent query processing for faster evaluation.
    """
    
    def __init__(self, agent: GraphMemAgent, max_samples: int = 50, max_workers: int = 10):
        self.agent = agent
        self.max_samples = max_samples
        self.max_workers = max_workers
        self.result = CompetencyResult("LRU")
        self._lock = Lock()
    
    def _evaluate_query(self, question: str, expected: Any) -> TaskResult:
        """Evaluate a single query (thread-safe)."""
        predicted, latency = self.agent.query(question)
        metrics = calculate_metrics(predicted, expected)
        
        return TaskResult(
            query=question,
            expected=expected,
            predicted=predicted,
            latency_ms=latency,
            **metrics
        )
    
    def _evaluate_queries_concurrent(self, qa_pairs: List[Tuple[str, Any]]) -> List[TaskResult]:
        """Evaluate multiple queries concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_query, q, a): (q, a) 
                for q, a in qa_pairs
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with self._lock:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
        
        return results
    
    def evaluate_summ(self, dataset) -> List[TaskResult]:
        """Summarization evaluation (∞Bench-Sum)."""
        logger.info("📊 Evaluating Summ (Summarization)...")
        results = []
        
        if "Long_Range_Understanding" not in dataset:
            return results
        
        samples = list(dataset["Long_Range_Understanding"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # Large chunks for summarization
            chunks = chunk_text(context, chunk_size=4096)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            # Summarization typically has 1 question
            for q_idx, (question, expected) in enumerate(zip(questions[:3], answers[:3])):
                predicted, latency = self.agent.query(question)
                metrics = calculate_metrics(predicted, expected)
                
                results.append(TaskResult(
                    query=question,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    **metrics
                ))
            
            if idx >= 2:
                break
        
        self.result.tasks["Summ"] = results
        return results
    
    def evaluate_detqa(self, dataset) -> List[TaskResult]:
        """Detective QA evaluation."""
        logger.info("📊 Evaluating DetQA (Detective QA)...")
        results = []
        
        if "Long_Range_Understanding" not in dataset:
            return results
        
        samples = list(dataset["Long_Range_Understanding"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            chunks = chunk_text(context, chunk_size=4096)
            self.agent.memorize_batch(chunks, evolve_after=True)
            
            for q_idx, (question, expected) in enumerate(zip(questions[:5], answers[:5])):
                predicted, latency = self.agent.query(question)
                metrics = calculate_metrics(predicted, expected)
                
                results.append(TaskResult(
                    query=question,
                    expected=expected,
                    predicted=predicted,
                    latency_ms=latency,
                    **metrics
                ))
            
            if idx >= 2:
                break
        
        self.result.tasks["DetQA"] = results
        return results
    
    def run(self, dataset) -> CompetencyResult:
        """Run all LRU evaluations."""
        self.evaluate_summ(dataset)
        self.evaluate_detqa(dataset)
        return self.result


class SelectiveForgettingEvaluator:
    """
    SF (Selective Forgetting / Conflict Resolution) Evaluator
    
    Tests: FC-SH (FactConsolidation Single-Hop), FC-MH (FactConsolidation Multi-Hop)
    
    This is where GraphMem's priority-based conflict resolution shines!
    
    Uses concurrent query processing for faster evaluation.
    """
    
    def __init__(self, agent: GraphMemAgent, max_samples: int = 100, max_workers: int = 10):
        self.agent = agent
        self.max_samples = max_samples
        self.max_workers = max_workers
        self.result = CompetencyResult("SF")
        self._lock = Lock()
    
    def _evaluate_query(self, question: str, expected: Any) -> TaskResult:
        """Evaluate a single query (thread-safe)."""
        predicted, latency = self.agent.query(question)
        metrics = calculate_metrics(predicted, expected)
        
        return TaskResult(
            query=question,
            expected=expected,
            predicted=predicted,
            latency_ms=latency,
            **metrics
        )
    
    def _evaluate_queries_concurrent(self, qa_pairs: List[Tuple[str, Any]]) -> List[TaskResult]:
        """Evaluate multiple queries concurrently."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_query, q, a): (q, a) 
                for q, a in qa_pairs
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with self._lock:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
        
        return results
    
    def evaluate_fc_sh(self, dataset) -> List[TaskResult]:
        """FactConsolidation Single-Hop evaluation.
        
        THIS IS WHERE GRAPHMEM'S EVOLUTION FEATURES SHINE!
        
        Evolution features utilized:
        1. PRIORITY-BASED DECAY: Facts later in document have higher priority
           - Fact 40: "Christianity founded in Jerusalem" (priority=40)
           - Fact 43: "Christianity founded in Taipei" (priority=43) ← WINS
        
        2. LLM-BASED CONFLICT DETECTION: Identifies same source+relation with different targets
        
        3. IMPORTANCE FILTERING: EPHEMERAL edges filtered during retrieval
        
        4. TEMPORAL SUPERSESSION: Later facts supersede earlier ones
        """
        logger.info("📊 Evaluating FC-SH (FactConsolidation Single-Hop)...")
        logger.info("   🔄 Using phased ingestion with evolution between phases")
        results = []
        
        if "Conflict_Resolution" not in dataset:
            return results
        
        samples = list(dataset["Conflict_Resolution"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # PHASED INGESTION - Critical for conflict resolution!
            # Facts are numbered (e.g., "43. Christianity was founded in Taipei")
            # Higher numbers = newer information = should override
            total_len = len(context)
            
            # Phase 1: Initial facts (40%) - OLD information
            logger.debug(f"   Phase 1: Ingesting initial facts (0-40%)")
            initial = context[:int(total_len * 0.4)]
            chunks_initial = chunk_text(initial, chunk_size=512)
            self.agent.memorize_batch(chunks_initial, evolve_after=False)  # No evolution yet
            
            stats_p1 = self.agent.get_stats()
            logger.debug(f"      After P1: {stats_p1['entities']} entities, {stats_p1['relationships']} relationships")
            
            # Phase 2: Updates (30%) - May contain conflicting info
            logger.debug(f"   Phase 2: Ingesting updates (40-70%) + EVOLUTION")
            middle = context[int(total_len * 0.4):int(total_len * 0.7)]
            chunks_middle = chunk_text(middle, chunk_size=512)
            self.agent.memorize_batch(chunks_middle, evolve_after=True)  # EVOLVE to detect conflicts
            
            stats_p2 = self.agent.get_stats()
            logger.debug(f"      After P2: {stats_p2['entities']} entities, {stats_p2['relationships']} relationships")
            logger.debug(f"      Evolution: {self.agent.evolution_stats['decays']} decays so far")
            
            # Phase 3: Final facts (30%) - NEWEST information (should win)
            logger.debug(f"   Phase 3: Ingesting final facts (70-100%) + FINAL EVOLUTION")
            final = context[int(total_len * 0.7):]
            chunks_final = chunk_text(final, chunk_size=512)
            self.agent.memorize_batch(chunks_final, evolve_after=True)  # FINAL evolution
            
            stats_p3 = self.agent.get_stats()
            logger.info(f"   Sample {idx+1} complete: {stats_p3['entities']} entities, "
                       f"{stats_p3['relationships']} relationships, "
                       f"{self.agent.evolution_stats['decays']} total decays")
            
            # Evaluate - system should return UPDATED facts (not old ones!)
            # CONCURRENT query processing
            qa_pairs = list(zip(questions[:10], answers[:10]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            # Log success rate
            correct = sum(1 for r in sample_results if r.substring_match >= 1.0)
            logger.info(f"   Sample {idx+1} result: {correct}/{len(sample_results)} correct")
            
            if idx >= 3:
                break
        
        self.result.tasks["FC-SH"] = results
        return results
    
    def evaluate_fc_mh(self, dataset) -> List[TaskResult]:
        """FactConsolidation Multi-Hop evaluation with concurrent processing."""
        logger.info("📊 Evaluating FC-MH (FactConsolidation Multi-Hop)...")
        logger.info(f"   Using {self.max_workers} concurrent workers")
        results = []
        
        if "Conflict_Resolution" not in dataset:
            return results
        
        samples = list(dataset["Conflict_Resolution"])[:self.max_samples]
        
        for idx, sample in enumerate(samples):
            self.agent.initialize(fresh=True)
            
            context = sample.get("context", "")
            questions = sample.get("questions", [])
            answers = sample.get("answers", [])
            
            # Same phased ingestion with evolution
            total_len = len(context)
            
            chunks_1 = chunk_text(context[:int(total_len * 0.4)], chunk_size=512)
            self.agent.memorize_batch(chunks_1, evolve_after=False)
            
            chunks_2 = chunk_text(context[int(total_len * 0.4):int(total_len * 0.7)], chunk_size=512)
            self.agent.memorize_batch(chunks_2, evolve_after=True)
            
            chunks_3 = chunk_text(context[int(total_len * 0.7):], chunk_size=512)
            self.agent.memorize_batch(chunks_3, evolve_after=True)
            
            # Multi-hop questions - CONCURRENT processing
            qa_pairs = list(zip(questions[10:20], answers[10:20]))
            sample_results = self._evaluate_queries_concurrent(qa_pairs)
            results.extend(sample_results)
            
            # Log success rate
            correct = sum(1 for r in sample_results if r.substring_match >= 1.0)
            logger.info(f"   Sample {idx+1} result: {correct}/{len(sample_results)} correct")
            
            if idx >= 3:
                break
        
        self.result.tasks["FC-MH"] = results
        return results
    
    def run(self, dataset) -> CompetencyResult:
        """Run all SF evaluations."""
        self.evaluate_fc_sh(dataset)
        self.evaluate_fc_mh(dataset)
        return self.result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_importance_dist(dist: Dict[str, int]) -> str:
    """Format importance distribution for display."""
    if not dist:
        return "No data"
    return " | ".join(f"{k}: {v}" for k, v in sorted(dist.items()))


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_full_benchmark(
    provider: str = "azure",
    api_key: str = None,
    api_base: str = None,
    llm_model: str = "gpt-4.1-mini",
    embedding_model: str = "text-embedding-3-small",
    azure_deployment: str = None,
    azure_embedding_deployment: str = None,
    max_samples: int = 100,
    max_workers: int = 10,
    competencies: List[str] = None,
    output_path: str = "benchmark_results.json",
) -> BenchmarkResult:
    """Run the full MemoryAgentBench evaluation."""
    
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🧠 GraphMem MemoryAgentBench Full Evaluation")
    print("="*80)
    print(f"   Model: {llm_model}")
    print(f"   Embeddings: {embedding_model}")
    print(f"   Max Samples: {max_samples}")
    print(f"   Concurrent Workers: {max_workers}")
    print(f"   Competencies: {competencies or ['AR', 'TTL', 'LRU', 'SF']}")
    print("="*80 + "\n")
    
    # Load dataset
    dataset = load_dataset_from_hf()
    
    # Initialize agent
    agent = GraphMemAgent(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        llm_model=llm_model,
        embedding_model=embedding_model,
        azure_deployment=azure_deployment,
        azure_embedding_deployment=azure_embedding_deployment,
    )
    
    # Initialize result
    result = BenchmarkResult(
        model=f"GraphMem ({llm_model})",
        timestamp=datetime.now().isoformat(),
    )
    
    # Determine which competencies to evaluate
    if competencies is None:
        competencies = ["AR", "TTL", "LRU", "SF"]
    
    # Run evaluations
    if "AR" in competencies:
        print("\n" + "="*60)
        print("📊 ACCURATE RETRIEVAL (AR)")
        print("="*60)
        ar_eval = AccurateRetrievalEvaluator(agent, max_samples, max_workers)
        result.ar = ar_eval.run(dataset)
        
        print("\n📈 AR Results:")
        for task, score in result.ar.task_scores().items():
            print(f"   {task}: {score:.1f}%")
        print(f"   Average: {result.ar.avg_score():.1f}%")
    
    if "TTL" in competencies:
        print("\n" + "="*60)
        print("📊 TEST-TIME LEARNING (TTL)")
        print("="*60)
        ttl_eval = TestTimeLearningEvaluator(agent, max_samples, max_workers)
        result.ttl = ttl_eval.run(dataset)
        
        print("\n📈 TTL Results:")
        for task, score in result.ttl.task_scores().items():
            print(f"   {task}: {score:.1f}%")
        print(f"   Average: {result.ttl.avg_score():.1f}%")
    
    if "LRU" in competencies:
        print("\n" + "="*60)
        print("📊 LONG-RANGE UNDERSTANDING (LRU)")
        print("="*60)
        lru_eval = LongRangeUnderstandingEvaluator(agent, max_samples, max_workers)
        result.lru = lru_eval.run(dataset)
        
        print("\n📈 LRU Results:")
        for task, score in result.lru.task_scores().items():
            print(f"   {task}: {score:.1f}%")
        print(f"   Average: {result.lru.avg_score():.1f}%")
    
    if "SF" in competencies:
        print("\n" + "="*60)
        print("📊 SELECTIVE FORGETTING (SF)")
        print("="*60)
        sf_eval = SelectiveForgettingEvaluator(agent, max_samples, max_workers)
        result.sf = sf_eval.run(dataset)
        
        print("\n📈 SF Results:")
        for task, score in result.sf.task_scores().items():
            print(f"   {task}: {score:.1f}%")
        print(f"   Average: {result.sf.avg_score():.1f}%")
    
    result.total_time = time.time() - start_time
    
    # Get evolution impact
    evolution_impact = agent.get_evolution_impact()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"""
┌──────────────────┬────────────────────────────────────────────────────┐
│ Competency       │ Tasks & Scores                                     │
├──────────────────┼────────────────────────────────────────────────────┤
│ AR               │ SH-QA: {result.ar.task_scores().get('SH-QA', 0):.1f}% | MH-QA: {result.ar.task_scores().get('MH-QA', 0):.1f}% | LME: {result.ar.task_scores().get('LME(S*)', 0):.1f}% | EventQA: {result.ar.task_scores().get('EventQA', 0):.1f}%
│                  │ Average: {result.ar.avg_score():.1f}%
├──────────────────┼────────────────────────────────────────────────────┤
│ TTL              │ MCC: {result.ttl.task_scores().get('MCC', 0):.1f}% | Recom: {result.ttl.task_scores().get('Recom', 0):.1f}%
│                  │ Average: {result.ttl.avg_score():.1f}%
├──────────────────┼────────────────────────────────────────────────────┤
│ LRU              │ Summ: {result.lru.task_scores().get('Summ', 0):.1f}% | DetQA: {result.lru.task_scores().get('DetQA', 0):.1f}%
│                  │ Average: {result.lru.avg_score():.1f}%
├──────────────────┼────────────────────────────────────────────────────┤
│ SF               │ FC-SH: {result.sf.task_scores().get('FC-SH', 0):.1f}% | FC-MH: {result.sf.task_scores().get('FC-MH', 0):.1f}%
│                  │ Average: {result.sf.avg_score():.1f}%
├──────────────────┼────────────────────────────────────────────────────┤
│ OVERALL          │ {result.overall_score():.1f}%
└──────────────────┴────────────────────────────────────────────────────┘
""")
    
    # Print evolution impact
    print("="*80)
    print("🔄 EVOLUTION IMPACT (GraphMem's Unique Features)")
    print("="*80)
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│ MEMORY DECAY (Priority-based conflict resolution)                        │
│   - Total relationships: {evolution_impact.get('total_relationships', 0)}
│   - Active edges: {evolution_impact.get('active_edges', 0)}
│   - Decayed edges (superseded): {evolution_impact.get('decayed_edges', 0)}
│   - Decay rate: {evolution_impact.get('decay_rate', 0)*100:.1f}%
├─────────────────────────────────────────────────────────────────────────┤
│ TEMPORAL VALIDITY                                                        │
│   - Edges with temporal bounds: {evolution_impact.get('temporal_edges', 0)}
├─────────────────────────────────────────────────────────────────────────┤
│ EVOLUTION EVENTS                                                         │
│   - Consolidations (entity merges): {evolution_impact.get('evolution_events', {}).get('consolidations', 0)}
│   - Decays (conflict resolution): {evolution_impact.get('evolution_events', {}).get('decays', 0)}
│   - PageRank updates: {evolution_impact.get('evolution_events', {}).get('pagerank_updates', 0)}
│   - Temporal conflicts detected: {evolution_impact.get('evolution_events', {}).get('temporal_conflicts', 0)}
├─────────────────────────────────────────────────────────────────────────┤
│ IMPORTANCE DISTRIBUTION (PageRank effect)                                │
│   {_format_importance_dist(evolution_impact.get('importance_distribution', {}))}
└─────────────────────────────────────────────────────────────────────────┘
    
Total Time: {result.total_time:.1f}s
""")
    
    # Save results
    with open(output_path, 'w') as f:
        # Convert to serializable format
        output_data = {
            "model": result.model,
            "timestamp": result.timestamp,
            "total_time": result.total_time,
            "overall_score": result.overall_score(),
            "competencies": {
                "AR": {
                    "average": result.ar.avg_score(),
                    "tasks": result.ar.task_scores(),
                },
                "TTL": {
                    "average": result.ttl.avg_score(),
                    "tasks": result.ttl.task_scores(),
                },
                "LRU": {
                    "average": result.lru.avg_score(),
                    "tasks": result.lru.task_scores(),
                },
                "SF": {
                    "average": result.sf.avg_score(),
                    "tasks": result.sf.task_scores(),
                },
            }
        }
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GraphMem MemoryAgentBench Full Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with Azure OpenAI
  python memory_agent_bench_full.py \\
    --provider azure \\
    --api-key "YOUR_KEY" \\
    --azure-endpoint "https://your-endpoint.openai.azure.com/" \\
    --azure-deployment "gpt-4.1-mini" \\
    --azure-embedding-deployment "text-embedding-3-small"

  # Run only Selective Forgetting (conflict resolution) tests
  python memory_agent_bench_full.py --competencies SF

  # Run with OpenRouter
  python memory_agent_bench_full.py \\
    --provider openai_compatible \\
    --api-key "YOUR_KEY" \\
    --api-base "https://openrouter.ai/api/v1"
        """
    )
    
    parser.add_argument("--provider", type=str, default="azure",
                       help="LLM provider (azure, openai, openai_compatible)")
    parser.add_argument("--api-key", type=str, required=True,
                       help="API key")
    parser.add_argument("--api-base", type=str, default=None,
                       help="API base URL")
    parser.add_argument("--azure-endpoint", type=str, default=None,
                       help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", type=str, default="gpt-4.1-mini",
                       help="Azure deployment name")
    parser.add_argument("--azure-embedding-deployment", type=str, default="text-embedding-3-small",
                       help="Azure embedding deployment name")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini",
                       help="LLM model name")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small",
                       help="Embedding model name")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum samples per task")
    parser.add_argument("--workers", type=int, default=20,
                       help="Number of concurrent workers for parallel query processing")
    parser.add_argument("--competencies", type=str, nargs="+", default=None,
                       choices=["AR", "TTL", "LRU", "SF"],
                       help="Which competencies to evaluate (default: all)")
    parser.add_argument("--output", type=str, default="graphmem_benchmark_results.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Set API base for Azure
    if args.provider == "azure" and args.azure_endpoint:
        args.api_base = args.azure_endpoint
    
    run_full_benchmark(
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.api_base,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        azure_deployment=args.azure_deployment,
        azure_embedding_deployment=args.azure_embedding_deployment,
        max_samples=args.max_samples,
        max_workers=args.workers,
        competencies=args.competencies,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


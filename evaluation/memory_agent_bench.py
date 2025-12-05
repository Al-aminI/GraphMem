#!/usr/bin/env python3
"""
MemoryAgentBench Evaluation for GraphMem

Properly evaluates GraphMem on 4 core memory competencies:

1. ACCURATE RETRIEVAL (AR) - 22 samples
   - Ingest full context (long documents)
   - Query for specific facts (single-hop and multi-hop)
   - Tests: precise information retrieval from massive histories

2. TEST-TIME LEARNING (TTL) - 6 samples  
   - Feed demonstrations/examples
   - Test if system learned patterns
   - Tests: in-context learning, pattern recognition

3. LONG-RANGE UNDERSTANDING (LRU) - 110 samples
   - Context is narrative (novel chapters)
   - Questions: "what happens next" based on previous events
   - Tests: global cognition, plot understanding, event prediction
   - Uses metadata.previous_events for context

4. CONFLICT RESOLUTION (CR) - 8 samples
   - Feed initial facts, then conflicting updates
   - Check if NEW information is returned (not old)
   - Tests: TEMPORAL VALIDITY, updating outdated info
   - This is where GraphMem's temporal features shine!

Dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
Paper: https://arxiv.org/pdf/2507.05257

Usage:
    python memory_agent_bench.py \
        --provider azure \
        --api-key "..." \
        --azure-endpoint "https://..." \
        --split Conflict_Resolution \
        --max-samples 2
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class QuestionResult:
    """Result for a single question."""
    question: str
    expected: str
    predicted: str
    correct: bool
    score: float  # 0-1
    latency_ms: float
    question_type: str = ""  # single_hop, multi_hop, temporal, etc.


@dataclass 
class EvolutionMetrics:
    """Metrics from memory evolution."""
    nodes_before: int = 0
    nodes_after: int = 0
    nodes_consolidated: int = 0
    relationships_before: int = 0
    relationships_after: int = 0
    clusters_formed: int = 0
    evolution_time_s: float = 0.0


@dataclass
class SplitResult:
    """Result for a dataset split."""
    split: str
    total_samples: int
    total_questions: int
    correct: int
    accuracy: float
    avg_score: float
    avg_latency_ms: float
    # Evolution
    evolution: EvolutionMetrics = field(default_factory=EvolutionMetrics)
    # Per-question results
    questions: List[QuestionResult] = field(default_factory=list)


# ============================================================================
# ANSWER CHECKER (LLM-as-Judge)
# ============================================================================
class AnswerChecker:
    """LLM-as-judge for answer correctness."""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
    
    def check(self, question: str, expected: str, predicted: str) -> Tuple[bool, float]:
        """Check if predicted answer is correct. Returns (is_correct, score)."""
        if not predicted or predicted.strip() == "":
            return False, 0.0
        
        expected_lower = expected.lower().strip()
        predicted_lower = predicted.lower().strip()
        
        # Exact match
        if expected_lower == predicted_lower:
            return True, 1.0
        
        # Expected contained in predicted
        if expected_lower in predicted_lower:
            return True, 1.0
        
        # Yes/No questions
        if expected_lower in ["yes", "no"]:
            first_words = predicted_lower.split()[:5]
            if expected_lower in first_words:
                return True, 1.0
        
        # LLM judge for complex cases
        prompt = f"""Evaluate if the predicted answer is correct.

Question: {question}
Expected: {expected}
Predicted: {predicted}

Is the predicted answer semantically correct? Consider:
- Core information must match
- Phrasing can differ
- Extra info is OK if core answer is present

Respond ONLY with JSON: {{"correct": true/false, "score": 0.0-1.0}}"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result.get("correct", False), float(result.get("score", 0.0))
        except Exception as e:
            logger.debug(f"LLM judge error: {e}")
        
        return False, 0.0


# ============================================================================
# MEMORY AGENT BENCH EVALUATOR
# ============================================================================
class MemoryAgentBenchEvaluator:
    """
    Proper evaluation for MemoryAgentBench dataset.
    
    Each split is evaluated differently based on what it tests.
    """
    
    SPLITS = ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"]
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: str = None,
        api_base: str = None,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
        turso_db_path: str = "memory_bench.db",
        debug: bool = False,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.azure_deployment = azure_deployment or llm_model
        self.azure_embedding_deployment = azure_embedding_deployment or embedding_model
        self.turso_db_path = turso_db_path
        self.debug = debug
        self.dataset = None
    
    def _load_dataset(self):
        """Load from HuggingFace."""
        try:
            from datasets import load_dataset
            logger.info("📥 Loading MemoryAgentBench from HuggingFace...")
            self.dataset = load_dataset("ai-hyz/MemoryAgentBench")
            for split in self.dataset:
                logger.info(f"   {split}: {len(self.dataset[split])} samples")
        except ImportError:
            raise ImportError("Install: pip install datasets")
    
    def _init_graphmem(self, fresh: bool = True):
        """Initialize GraphMem instance."""
        from graphmem import GraphMem, MemoryConfig
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        if fresh and os.path.exists(self.turso_db_path):
            os.remove(self.turso_db_path)
        
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
        
        return GraphMem(config)
    
    def _init_checker(self):
        """Initialize answer checker."""
        from graphmem.llm.providers import get_llm_provider
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        llm = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,
        )
        return AnswerChecker(llm)
    
    # ========================================================================
    # SPLIT-SPECIFIC EVALUATION METHODS
    # ========================================================================
    
    def eval_accurate_retrieval(self, sample: Dict, gm, checker: AnswerChecker) -> List[QuestionResult]:
        """
        ACCURATE RETRIEVAL (AR)
        
        Task: Find specific information from massive context
        - Ingest full context
        - Query for facts (single-hop and multi-hop)
        """
        context = sample["context"]
        questions = sample["questions"]
        answers = sample["answers"]
        
        results = []
        
        # Ingest full context in chunks with CONTINUOUS EVOLUTION
        logger.info("      📄 Ingesting context with CONTINUOUS evolution...")
        chunk_size = 8000
        chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
        
        # Split into batches for continuous evolution
        batch_size = max(5, len(chunks) // 4)  # ~4 evolution cycles
        all_docs = [{"id": f"chunk_{i}", "content": chunk} for i, chunk in enumerate(chunks[:100])]
        
        for batch_idx in range(0, len(all_docs), batch_size):
            batch = all_docs[batch_idx:batch_idx + batch_size]
            
            try:
                gm.ingest_batch(
                    documents=batch,
                    max_workers=20,
                    show_progress=False,
                    aggressive=True,
                )
            except Exception as e:
                logger.warning(f"Batch ingestion error: {e}")
            
            # Evolve after each batch (except first) - CONTINUOUS LEARNING
            if batch_idx > 0:
                try:
                    evolution_result = gm.evolve()
                    events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
                    consolidations = sum(1 for e in events if 'consolidat' in str(e.evolution_type).lower())
                    decays = sum(1 for e in events if 'decay' in str(e.evolution_type).lower())
                    logger.info(f"         Batch {batch_idx//batch_size + 1}: {len(batch)} docs → {consolidations} consolidations, {decays} decays")
                except Exception as e:
                    logger.debug(f"Evolution error: {e}")
        
        # Final evolution after all ingestion
        logger.info("      🔄 Final evolution...")
        try:
            gm.evolve()
        except Exception as e:
            logger.debug(f"Final evolution error: {e}")
        
        # Query each question
        for i, (q, a_list) in enumerate(zip(questions, answers)):
            expected = a_list[0] if isinstance(a_list, list) else a_list
            
            start = time.time()
            try:
                response = gm.query(q)
                predicted = response.answer
            except Exception as e:
                predicted = f"Error: {e}"
            latency = (time.time() - start) * 1000
            
            is_correct, score = checker.check(q, expected, predicted)
            
            results.append(QuestionResult(
                question=q,
                expected=expected,
                predicted=predicted,
                correct=is_correct,
                score=score,
                latency_ms=latency,
                question_type="retrieval",
            ))
            
            if self.debug:
                status = "✅" if is_correct else "❌"
                logger.info(f"      {status} Q{i+1}: {q[:60]}...")
                logger.info(f"         Expected: {expected[:60]}")
                logger.info(f"         Got: {predicted[:60]}")
        
        return results
    
    def eval_test_time_learning(self, sample: Dict, gm, checker: AnswerChecker) -> List[QuestionResult]:
        """
        TEST-TIME LEARNING (TTL)
        
        Task: Learn patterns from examples and apply them
        - Feed demonstrations first
        - Test if system learned the pattern
        """
        context = sample["context"]
        questions = sample["questions"]
        answers = sample["answers"]
        metadata = sample.get("metadata", {})
        
        results = []
        
        # For TTL, context often contains demonstrations
        # Ingest with CONTINUOUS evolution to learn patterns incrementally
        logger.info("      📚 Learning from demonstrations with CONTINUOUS evolution...")
        
        chunk_size = 5000
        chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
        
        # Split into batches for continuous learning
        batch_size = max(3, len(chunks) // 3)
        all_docs = [{"id": f"demo_{i}", "content": chunk} for i, chunk in enumerate(chunks[:50])]
        
        for batch_idx in range(0, len(all_docs), batch_size):
            batch = all_docs[batch_idx:batch_idx + batch_size]
            
            try:
                gm.ingest_batch(
                    documents=batch,
                    max_workers=20,
                    show_progress=False,
                    aggressive=True,
                )
            except Exception as e:
                logger.warning(f"Demo ingestion error: {e}")
            
            # Evolve after each batch (except first) - pattern consolidation
            if batch_idx > 0:
                try:
                    evolution_result = gm.evolve()
                    events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
                    logger.info(f"         Batch {batch_idx//batch_size + 1}: {len(events)} evolution events")
                except:
                    pass
        
        # Final evolution
        logger.info("      🔄 Final pattern consolidation...")
        try:
            gm.evolve()
        except:
            pass
        
        # Test questions
        for i, (q, a_list) in enumerate(zip(questions, answers)):
            expected = a_list[0] if isinstance(a_list, list) else a_list
            
            start = time.time()
            try:
                response = gm.query(q)
                predicted = response.answer
            except Exception as e:
                predicted = f"Error: {e}"
            latency = (time.time() - start) * 1000
            
            is_correct, score = checker.check(q, expected, predicted)
            
            results.append(QuestionResult(
                question=q,
                expected=expected,
                predicted=predicted,
                correct=is_correct,
                score=score,
                latency_ms=latency,
                question_type="learning",
            ))
            
            if self.debug:
                status = "✅" if is_correct else "❌"
                logger.info(f"      {status} Q{i+1}: {q[:60]}...")
        
        return results
    
    def eval_long_range_understanding(self, sample: Dict, gm, checker: AnswerChecker) -> List[QuestionResult]:
        """
        LONG-RANGE UNDERSTANDING (LRU)
        
        Task: Understand narrative flow, predict what happens next
        - Context is narrative (novel chapters)
        - Questions use previous_events from metadata
        - Tests: global understanding, event sequences
        
        This is where GraphMem's relationship tracking shines!
        """
        context = sample["context"]
        questions = sample["questions"]
        answers = sample["answers"]
        metadata = sample.get("metadata", {})
        previous_events = metadata.get("previous_events", [])
        keypoints = metadata.get("keypoints", [])
        
        results = []
        
        # Ingest the narrative with CONTINUOUS evolution
        logger.info("      📖 Ingesting narrative with CONTINUOUS evolution...")
        chunk_size = 6000
        chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
        
        # Split into batches for continuous evolution
        batch_size = max(5, len(chunks) // 4)
        all_docs = [{"id": f"narrative_{i}", "content": chunk} for i, chunk in enumerate(chunks[:80])]
        
        for batch_idx in range(0, len(all_docs), batch_size):
            batch = all_docs[batch_idx:batch_idx + batch_size]
            
            try:
                gm.ingest_batch(
                    documents=batch,
                    max_workers=20,
                    show_progress=False,
                    aggressive=True,
                )
            except Exception as e:
                logger.warning(f"Narrative ingestion error: {e}")
            
            # Evolve after each batch (except first) - builds narrative graph
            if batch_idx > 0:
                try:
                    evolution_result = gm.evolve()
                    events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
                    logger.info(f"         Chapter batch {batch_idx//batch_size + 1}: {len(events)} evolution events")
                except:
                    pass
        
        # Final evolution to build complete narrative graph
        logger.info("      🔄 Building complete narrative graph...")
        try:
            gm.evolve()
        except:
            pass
        
        # If we have keypoints, ingest them and evolve again
        if keypoints:
            logger.info(f"      📌 Processing {len(keypoints)} keypoints...")
            kp_docs = [{"id": f"keypoint_{i}", "content": str(kp)} for i, kp in enumerate(keypoints[:20])]
            try:
                gm.ingest_batch(
                    documents=kp_docs,
                    max_workers=20,
                    show_progress=False,
                    aggressive=True,
                )
                # Evolve after keypoints
                gm.evolve()
            except:
                pass
        
        # Query with previous_events context
        for i, (q, a_list) in enumerate(zip(questions, answers)):
            expected = a_list[0] if isinstance(a_list, list) else a_list
            
            # LRU questions often include "These are the events that have already occurred"
            # The question itself contains the context
            
            start = time.time()
            try:
                response = gm.query(q)
                predicted = response.answer
            except Exception as e:
                predicted = f"Error: {e}"
            latency = (time.time() - start) * 1000
            
            is_correct, score = checker.check(q, expected, predicted)
            
            results.append(QuestionResult(
                question=q,
                expected=expected,
                predicted=predicted,
                correct=is_correct,
                score=score,
                latency_ms=latency,
                question_type="understanding",
            ))
            
            if self.debug:
                status = "✅" if is_correct else "❌"
                logger.info(f"      {status} Q{i+1}: {q[:80]}...")
                logger.info(f"         Expected: {expected[:80]}")
        
        return results
    
    def eval_conflict_resolution(self, sample: Dict, gm, checker: AnswerChecker) -> List[QuestionResult]:
        """
        CONFLICT RESOLUTION (CR)
        
        Task: Handle conflicting/updated information
        - Ingest initial facts
        - Ingest conflicting updates
        - Query and verify NEW information is returned
        
        THIS IS WHERE GRAPHMEM'S TEMPORAL VALIDITY SHINES!
        - valid_from / valid_until on relationships
        - Superseding old facts with new ones
        - LLM-based decay identifies outdated relationships
        """
        context = sample["context"]
        questions = sample["questions"]
        answers = sample["answers"]
        
        results = []
        
        logger.info("      ⚡ CONFLICT RESOLUTION - Testing temporal validity with CONTINUOUS evolution...")
        
        # Split context into multiple phases to simulate temporal updates
        total_len = len(context)
        chunk_size = 5000
        
        # Phase 1: Ingest first 40% (initial facts) - NO evolution yet
        initial_context = context[:int(total_len * 0.4)]
        logger.info("      📝 Phase 1: Ingesting INITIAL facts (baseline)...")
        
        initial_chunks = [initial_context[i:i+chunk_size] for i in range(0, len(initial_context), chunk_size)]
        initial_docs = [{"id": f"initial_{i}", "content": chunk} for i, chunk in enumerate(initial_chunks[:20])]
        
        try:
            gm.ingest_batch(
                documents=initial_docs,
                max_workers=20,
                show_progress=False,
                aggressive=True,
            )
        except Exception as e:
            logger.warning(f"Initial facts ingestion error: {e}")
        
        # DON'T evolve after first ingestion - establish baseline
        nodes_initial = len(gm._memory.nodes)
        edges_initial = len(gm._memory.edges)
        logger.info(f"         Initial state: {nodes_initial} entities, {edges_initial} relationships")
        
        # Phase 2: Ingest middle 30% (some updates) - EVOLVE to consolidate
        middle_context = context[int(total_len * 0.4):int(total_len * 0.7)]
        logger.info("      🔄 Phase 2: Ingesting UPDATES + evolving...")
        
        middle_chunks = [middle_context[i:i+chunk_size] for i in range(0, len(middle_context), chunk_size)]
        middle_docs = [{"id": f"middle_{i}", "content": chunk} for i, chunk in enumerate(middle_chunks[:15])]
        
        try:
            gm.ingest_batch(
                documents=middle_docs,
                max_workers=20,
                show_progress=False,
                aggressive=True,
            )
        except Exception as e:
            logger.warning(f"Middle ingestion error: {e}")
        
        # EVOLVE - should start detecting temporal conflicts
        try:
            evolution_result = gm.evolve()
            events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
            consolidations = sum(1 for e in events if 'consolidat' in str(e.evolution_type).lower())
            decays = sum(1 for e in events if 'decay' in str(e.evolution_type).lower())
            logger.info(f"         Evolution 1: {consolidations} consolidations, {decays} decays")
        except:
            pass
        
        # Phase 3: Ingest final 30% (latest facts) - EVOLVE for conflict resolution
        final_context = context[int(total_len * 0.7):]
        logger.info("      ⚡ Phase 3: Ingesting LATEST facts + CONFLICT resolution...")
        
        final_chunks = [final_context[i:i+chunk_size] for i in range(0, len(final_context), chunk_size)]
        final_docs = [{"id": f"final_{i}", "content": chunk} for i, chunk in enumerate(final_chunks[:15])]
        
        try:
            gm.ingest_batch(
                documents=final_docs,
                max_workers=20,
                show_progress=False,
                aggressive=True,
            )
        except Exception as e:
            logger.warning(f"Final ingestion error: {e}")
        
        # FINAL EVOLUTION - LLM-based decay should identify and mark outdated relationships
        try:
            evolution_result = gm.evolve()
            events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
            consolidations = sum(1 for e in events if 'consolidat' in str(e.evolution_type).lower())
            decays = sum(1 for e in events if 'decay' in str(e.evolution_type).lower())
            logger.info(f"         Evolution 2: {consolidations} consolidations, {decays} decays")
        except:
            pass
        
        nodes_final = len(gm._memory.nodes)
        edges_final = len(gm._memory.edges)
        logger.info(f"         Final state: {nodes_final} entities, {edges_final} relationships")
        logger.info(f"         Change: {nodes_final - nodes_initial:+d} entities, {edges_final - edges_initial:+d} relationships")
        
        # Query - should return UPDATED information, not old
        for i, (q, a_list) in enumerate(zip(questions, answers)):
            expected = a_list[0] if isinstance(a_list, list) else a_list
            
            start = time.time()
            try:
                response = gm.query(q)
                predicted = response.answer
            except Exception as e:
                predicted = f"Error: {e}"
            latency = (time.time() - start) * 1000
            
            is_correct, score = checker.check(q, expected, predicted)
            
            results.append(QuestionResult(
                question=q,
                expected=expected,
                predicted=predicted,
                correct=is_correct,
                score=score,
                latency_ms=latency,
                question_type="conflict_resolution",
            ))
            
            if self.debug:
                status = "✅" if is_correct else "❌"
                logger.info(f"      {status} Q{i+1}: {q[:60]}...")
                logger.info(f"         Expected (NEW): {expected[:60]}")
                logger.info(f"         Got: {predicted[:60]}")
        
        return results
    
    # ========================================================================
    # MAIN EVALUATION
    # ========================================================================
    
    def run_split(self, split: str, max_samples: int = None, max_questions: int = None) -> SplitResult:
        """Run evaluation on a specific split."""
        if self.dataset is None:
            self._load_dataset()
        
        if split not in self.dataset:
            raise ValueError(f"Invalid split: {split}")
        
        samples = list(self.dataset[split])
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"\n{'='*80}")
        print(f"📊 EVALUATING: {split}")
        print(f"   Samples: {len(samples)}")
        print(f"{'='*80}")
        
        checker = self._init_checker()
        all_results = []
        total_evolution = EvolutionMetrics()
        
        # Select evaluation method based on split
        eval_method = {
            "Accurate_Retrieval": self.eval_accurate_retrieval,
            "Test_Time_Learning": self.eval_test_time_learning,
            "Long_Range_Understanding": self.eval_long_range_understanding,
            "Conflict_Resolution": self.eval_conflict_resolution,
        }.get(split, self.eval_accurate_retrieval)
        
        for idx, sample in enumerate(samples):
            print(f"\n--- Sample {idx+1}/{len(samples)} ---")
            print(f"   Context: {len(sample['context']):,} chars")
            print(f"   Questions: {len(sample['questions'])}")
            
            # Fresh GraphMem for each sample
            gm = self._init_graphmem(fresh=True)
            
            # Get stats before
            try:
                nodes_before = len(gm._memory.nodes)
                rels_before = len(gm._memory.edges)
            except:
                nodes_before, rels_before = 0, 0
            
            # Run split-specific evaluation
            start = time.time()
            questions_to_eval = sample["questions"]
            answers_to_eval = sample["answers"]
            
            if max_questions:
                sample = dict(sample)
                sample["questions"] = questions_to_eval[:max_questions]
                sample["answers"] = answers_to_eval[:max_questions]
            
            results = eval_method(sample, gm, checker)
            eval_time = time.time() - start
            
            # Get stats after
            try:
                nodes_after = len(gm._memory.nodes)
                rels_after = len(gm._memory.edges)
                clusters = len(gm._memory.clusters) if hasattr(gm._memory, 'clusters') else 0
            except:
                nodes_after, rels_after, clusters = 0, 0, 0
            
            # Accumulate
            all_results.extend(results)
            total_evolution.nodes_before += nodes_before
            total_evolution.nodes_after += nodes_after
            total_evolution.relationships_before += rels_before
            total_evolution.relationships_after += rels_after
            total_evolution.clusters_formed += clusters
            total_evolution.evolution_time_s += eval_time
            
            # Print sample summary
            correct = sum(1 for r in results if r.correct)
            print(f"   ✅ Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
            print(f"   🧠 Entities: {nodes_before} → {nodes_after}")
            print(f"   🔗 Relationships: {rels_before} → {rels_after}")
        
        # Aggregate
        total_correct = sum(1 for r in all_results if r.correct)
        total_questions = len(all_results)
        
        result = SplitResult(
            split=split,
            total_samples=len(samples),
            total_questions=total_questions,
            correct=total_correct,
            accuracy=total_correct / total_questions if total_questions else 0,
            avg_score=sum(r.score for r in all_results) / len(all_results) if all_results else 0,
            avg_latency_ms=sum(r.latency_ms for r in all_results) / len(all_results) if all_results else 0,
            evolution=total_evolution,
            questions=all_results,
        )
        
        self._print_split_results(result)
        return result
    
    def _print_split_results(self, result: SplitResult):
        """Print results for a split."""
        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {result.split}")
        print(f"{'='*80}")
        
        print(f"\n🎯 ACCURACY")
        print(f"   Correct: {result.correct}/{result.total_questions} ({result.accuracy:.1%})")
        print(f"   Avg Score: {result.avg_score:.2f}")
        print(f"   Avg Latency: {result.avg_latency_ms:.0f}ms")
        
        print(f"\n🧠 MEMORY EVOLUTION")
        e = result.evolution
        print(f"   Entities: {e.nodes_before} → {e.nodes_after}")
        print(f"   Relationships: {e.relationships_before} → {e.relationships_after}")
        print(f"   Clusters: {e.clusters_formed}")
        print(f"   Time: {e.evolution_time_s:.1f}s")
        
        # Deduplication ratio
        if e.nodes_before > 0:
            dedup = 1 - (e.nodes_after / e.nodes_before)
            print(f"   Deduplication: {dedup:.1%} reduction")
        
        print(f"{'='*80}")
    
    def run_all(self, max_samples: int = 2, max_questions: int = 10) -> Dict[str, SplitResult]:
        """Run all splits."""
        if self.dataset is None:
            self._load_dataset()
        
        results = {}
        for split in self.SPLITS:
            if split in self.dataset:
                results[split] = self.run_split(split, max_samples, max_questions)
        
        # Overall summary
        self._print_overall(results)
        return results
    
    def _print_overall(self, results: Dict[str, SplitResult]):
        """Print overall summary."""
        print(f"\n{'='*80}")
        print("📊 OVERALL SUMMARY: MemoryAgentBench")
        print(f"{'='*80}")
        
        print(f"\n{'Split':<30} {'Accuracy':>15} {'Avg Score':>15}")
        print("-" * 60)
        
        total_correct = 0
        total_questions = 0
        
        for split, r in results.items():
            print(f"{split:<30} {r.accuracy:>14.1%} {r.avg_score:>15.2f}")
            total_correct += r.correct
            total_questions += r.total_questions
        
        print("-" * 60)
        overall_acc = total_correct / total_questions if total_questions else 0
        print(f"{'OVERALL':<30} {overall_acc:>14.1%}")
        print(f"{'='*80}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MemoryAgentBench Evaluation")
    
    parser.add_argument("--provider", default="azure")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--azure-endpoint", help="Azure endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4.1-mini")
    parser.add_argument("--azure-embedding-deployment", default="text-embedding-3-small")
    
    parser.add_argument("--split", default="all", help="Split to run or 'all'")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--turso-db", default="memory_bench.db")
    
    args = parser.parse_args()
    
    evaluator = MemoryAgentBenchEvaluator(
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.azure_endpoint,
        llm_model=args.azure_deployment,
        embedding_model=args.azure_embedding_deployment,
        azure_deployment=args.azure_deployment,
        azure_embedding_deployment=args.azure_embedding_deployment,
        turso_db_path=args.turso_db,
        debug=args.debug,
    )
    
    if args.split == "all":
        evaluator.run_all(args.max_samples, args.max_questions)
    else:
        evaluator.run_split(args.split, args.max_samples, args.max_questions)


if __name__ == "__main__":
    main()

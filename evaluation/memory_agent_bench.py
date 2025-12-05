#!/usr/bin/env python3
"""
MemoryAgentBench Evaluation for GraphMem

Evaluates GraphMem on the MemoryAgentBench dataset which tests 4 core memory competencies:
1. Accurate Retrieval (AR) - Locating required information from massive histories
2. Test-Time Learning (TTL) - Learning new skills during interactions  
3. Long-Range Understanding (LRU) - Forming global cognition from long conversations
4. Conflict Resolution (CR) - Identifying and updating outdated information

Dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
Paper: https://arxiv.org/pdf/2507.05257

Usage:
    python memory_agent_bench.py \
        --provider azure \
        --api-key "..." \
        --azure-endpoint "https://..." \
        --azure-deployment "gpt-4.1-mini" \
        --azure-embedding-deployment "text-embedding-3-small" \
        --split Accurate_Retrieval \
        --max-samples 5 \
        --max-questions 20
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
from typing import Dict, List, Optional, Tuple
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
    latency_ms: float
    context_tokens: int = 0


@dataclass
class SampleResult:
    """Result for a single sample (context + multiple questions)."""
    sample_id: int
    split: str
    num_questions: int
    num_correct: int
    accuracy: float
    questions: List[QuestionResult] = field(default_factory=list)
    ingestion_time_s: float = 0.0
    total_entities: int = 0
    total_relationships: int = 0


@dataclass
class BenchmarkResults:
    """Overall benchmark results."""
    split: str
    total_samples: int
    total_questions: int
    total_correct: int
    accuracy: float
    avg_latency_ms: float
    total_ingestion_time_s: float
    avg_entities_per_sample: float
    avg_relationships_per_sample: float
    samples: List[SampleResult] = field(default_factory=list)


# ============================================================================
# ANSWER CHECKER
# ============================================================================
class AnswerChecker:
    """LLM-as-judge for answer correctness."""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
    
    def check(self, question: str, expected: str, predicted: str) -> Tuple[bool, float]:
        """
        Check if predicted answer is correct using LLM-as-judge.
        Returns (is_correct, score 0-1).
        """
        if not predicted or predicted.strip() == "":
            return False, 0.0
        
        # Normalize for simple cases
        expected_lower = expected.lower().strip()
        predicted_lower = predicted.lower().strip()
        
        # Exact match
        if expected_lower == predicted_lower:
            return True, 1.0
        
        # Check if expected is contained in predicted
        if expected_lower in predicted_lower:
            return True, 1.0
        
        # For yes/no questions
        if expected_lower in ["yes", "no"]:
            if expected_lower in predicted_lower.split()[:5]:
                return True, 1.0
        
        # LLM-as-judge for complex cases
        prompt = f"""You are evaluating if a predicted answer is correct.

Question: {question}

Expected Answer: {expected}

Predicted Answer: {predicted}

Is the predicted answer semantically equivalent to or contains the expected answer?
Consider:
- The core information must match
- Phrasing can differ
- Extra information is OK if the core answer is present

Respond with ONLY a JSON object:
{{"correct": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages, temperature=0.0)
            
            # Parse JSON
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("correct", False), float(result.get("score", 0.0))
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
        
        return False, 0.0


# ============================================================================
# NAIVE RAG BASELINE
# ============================================================================
class NaiveRAGBaseline:
    """Simple chunk-and-retrieve RAG for comparison."""
    
    def __init__(self, llm_provider, embedding_provider, chunk_size: int = 1000):
        self.llm = llm_provider
        self.embedder = embedding_provider
        self.chunk_size = chunk_size
        self.chunks: List[Dict] = []
        self.embeddings: List[List[float]] = []
    
    def clear(self):
        """Clear all stored chunks."""
        self.chunks = []
        self.embeddings = []
    
    def ingest(self, content: str):
        """Chunk and embed content."""
        import numpy as np
        
        # Simple word-based chunking
        words = content.split()
        chunk_words = []
        
        for word in words:
            chunk_words.append(word)
            if len(' '.join(chunk_words)) >= self.chunk_size:
                chunk_content = ' '.join(chunk_words)
                try:
                    embedding = self.embedder.embed_text(chunk_content)
                    self.chunks.append({"content": chunk_content})
                    self.embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
                chunk_words = []
        
        # Remaining
        if chunk_words:
            chunk_content = ' '.join(chunk_words)
            try:
                embedding = self.embedder.embed_text(chunk_content)
                self.chunks.append({"content": chunk_content})
                self.embeddings.append(embedding)
            except Exception as e:
                pass
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Retrieve and answer."""
        import numpy as np
        
        if not self.embeddings:
            return "No context available."
        
        # Embed query
        try:
            query_emb = self.embedder.embed_text(question)
        except:
            return "Embedding failed."
        
        # Find top-k
        similarities = []
        for i, chunk_emb in enumerate(self.embeddings):
            a = np.array(query_emb)
            b = np.array(chunk_emb)
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build context
        context = "\n\n".join([self.chunks[i]["content"] for i, _ in similarities[:top_k]])
        
        # Generate answer
        prompt = f"""Answer the question based on the context. Be concise and direct.

Context:
{context[:8000]}

Question: {question}

Answer:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            return self.llm.chat(messages)
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# MEMORY AGENT BENCH EVALUATOR
# ============================================================================
class MemoryAgentBenchEvaluator:
    """
    Evaluator for MemoryAgentBench dataset.
    
    Tests GraphMem on 4 core memory competencies:
    - Accurate Retrieval (AR)
    - Test-Time Learning (TTL)
    - Long-Range Understanding (LRU)
    - Conflict Resolution (CR)
    """
    
    SPLITS = [
        "Accurate_Retrieval",
        "Test_Time_Learning", 
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]
    
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
        self.graphmem_results: Dict[str, BenchmarkResults] = {}
        self.naive_results: Dict[str, BenchmarkResults] = {}
    
    def _load_dataset(self):
        """Load MemoryAgentBench from HuggingFace."""
        try:
            from datasets import load_dataset
            logger.info("Loading MemoryAgentBench dataset from HuggingFace...")
            self.dataset = load_dataset("ai-hyz/MemoryAgentBench")
            logger.info(f"Dataset loaded with splits: {list(self.dataset.keys())}")
            for split in self.dataset.keys():
                logger.info(f"  {split}: {len(self.dataset[split])} samples")
        except ImportError:
            logger.error("Install datasets: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _init_graphmem(self):
        """Initialize GraphMem."""
        from graphmem import GraphMem, MemoryConfig
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        # Clean up existing DB
        if os.path.exists(self.turso_db_path):
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
    
    def _init_naive_rag(self):
        """Initialize Naive RAG baseline."""
        from graphmem.llm.providers import get_llm_provider
        from graphmem.llm.embeddings import get_embedding_provider
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        llm = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,
        )
        
        embedder = get_embedding_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.embedding_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_embedding_deployment,
        )
        
        return NaiveRAGBaseline(llm, embedder)
    
    def _init_answer_checker(self):
        """Initialize LLM-as-judge."""
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
    
    def run_split(
        self,
        split: str,
        max_samples: int = None,
        max_questions: int = None,
        compare_naive: bool = True,
    ) -> Tuple[BenchmarkResults, Optional[BenchmarkResults]]:
        """
        Run evaluation on a specific split.
        
        Args:
            split: Dataset split (Accurate_Retrieval, Test_Time_Learning, etc.)
            max_samples: Max samples to evaluate (None = all)
            max_questions: Max questions per sample (None = all)
            compare_naive: Whether to also run NaiveRAG baseline
        """
        if self.dataset is None:
            self._load_dataset()
        
        if split not in self.dataset:
            raise ValueError(f"Invalid split: {split}. Available: {list(self.dataset.keys())}")
        
        data = self.dataset[split]
        samples = list(data)
        
        if max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 EVALUATING: {split}")
        logger.info(f"   Samples: {len(samples)}")
        logger.info(f"   Max questions per sample: {max_questions or 'all'}")
        logger.info(f"{'='*80}")
        
        # Initialize
        answer_checker = self._init_answer_checker()
        
        # Results
        gm_sample_results = []
        naive_sample_results = []
        
        for sample_idx, sample in enumerate(samples):
            context = sample["context"]
            questions = sample["questions"]
            answers = sample["answers"]
            
            if max_questions:
                questions = questions[:max_questions]
                answers = answers[:max_questions]
            
            logger.info(f"\n--- Sample {sample_idx + 1}/{len(samples)} ---")
            logger.info(f"   Context length: {len(context):,} chars")
            logger.info(f"   Questions: {len(questions)}")
            
            # =============== GRAPHMEM ===============
            logger.info(f"\n   🧠 GraphMem:")
            
            # Fresh GraphMem for each sample
            gm = self._init_graphmem()
            
            # Ingest context
            ingest_start = time.time()
            try:
                # Split large context into chunks for ingestion
                chunk_size = 10000
                chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
                
                for i, chunk in enumerate(chunks[:50]):  # Limit to first 50 chunks
                    gm.ingest(chunk)
                    if (i + 1) % 10 == 0:
                        logger.info(f"      Ingested {i+1}/{min(len(chunks), 50)} chunks...")
                
            except Exception as e:
                logger.warning(f"      Ingestion error: {e}")
            
            ingest_time = time.time() - ingest_start
            logger.info(f"      Ingestion: {ingest_time:.1f}s")
            
            # Get memory stats
            try:
                memory = gm._memory
                num_entities = len(memory.nodes)
                num_relationships = len(memory.edges)
                logger.info(f"      Entities: {num_entities}, Relationships: {num_relationships}")
            except:
                num_entities = 0
                num_relationships = 0
            
            # Query
            gm_questions = []
            gm_correct = 0
            
            for q_idx, (q, a_list) in enumerate(zip(questions, answers)):
                # Handle multiple acceptable answers
                expected = a_list[0] if isinstance(a_list, list) else a_list
                
                q_start = time.time()
                try:
                    response = gm.query(q)
                    predicted = response.answer
                except Exception as e:
                    predicted = f"Error: {e}"
                q_time = (time.time() - q_start) * 1000
                
                # Check correctness
                is_correct, score = answer_checker.check(q, expected, predicted)
                if is_correct:
                    gm_correct += 1
                
                gm_questions.append(QuestionResult(
                    question=q,
                    expected=expected,
                    predicted=predicted,
                    correct=is_correct,
                    latency_ms=q_time,
                ))
                
                if self.debug:
                    status = "✅" if is_correct else "❌"
                    logger.info(f"      {status} Q{q_idx+1}: {q[:50]}...")
                    logger.info(f"         Expected: {expected[:50]}...")
                    logger.info(f"         Got: {predicted[:50]}...")
            
            gm_accuracy = gm_correct / len(questions) if questions else 0
            logger.info(f"      Accuracy: {gm_accuracy:.1%} ({gm_correct}/{len(questions)})")
            
            gm_sample_results.append(SampleResult(
                sample_id=sample_idx,
                split=split,
                num_questions=len(questions),
                num_correct=gm_correct,
                accuracy=gm_accuracy,
                questions=gm_questions,
                ingestion_time_s=ingest_time,
                total_entities=num_entities,
                total_relationships=num_relationships,
            ))
            
            # =============== NAIVE RAG ===============
            if compare_naive:
                logger.info(f"\n   📚 NaiveRAG:")
                
                naive = self._init_naive_rag()
                
                # Ingest
                naive_start = time.time()
                naive.ingest(context[:500000])  # Limit context size
                naive_ingest_time = time.time() - naive_start
                logger.info(f"      Ingestion: {naive_ingest_time:.1f}s ({len(naive.chunks)} chunks)")
                
                # Query
                naive_questions = []
                naive_correct = 0
                
                for q_idx, (q, a_list) in enumerate(zip(questions, answers)):
                    expected = a_list[0] if isinstance(a_list, list) else a_list
                    
                    q_start = time.time()
                    predicted = naive.query(q)
                    q_time = (time.time() - q_start) * 1000
                    
                    is_correct, score = answer_checker.check(q, expected, predicted)
                    if is_correct:
                        naive_correct += 1
                    
                    naive_questions.append(QuestionResult(
                        question=q,
                        expected=expected,
                        predicted=predicted,
                        correct=is_correct,
                        latency_ms=q_time,
                    ))
                
                naive_accuracy = naive_correct / len(questions) if questions else 0
                logger.info(f"      Accuracy: {naive_accuracy:.1%} ({naive_correct}/{len(questions)})")
                
                naive_sample_results.append(SampleResult(
                    sample_id=sample_idx,
                    split=split,
                    num_questions=len(questions),
                    num_correct=naive_correct,
                    accuracy=naive_accuracy,
                    questions=naive_questions,
                    ingestion_time_s=naive_ingest_time,
                ))
        
        # Aggregate results
        gm_total_q = sum(s.num_questions for s in gm_sample_results)
        gm_total_correct = sum(s.num_correct for s in gm_sample_results)
        gm_all_latencies = [q.latency_ms for s in gm_sample_results for q in s.questions]
        
        gm_results = BenchmarkResults(
            split=split,
            total_samples=len(gm_sample_results),
            total_questions=gm_total_q,
            total_correct=gm_total_correct,
            accuracy=gm_total_correct / gm_total_q if gm_total_q else 0,
            avg_latency_ms=sum(gm_all_latencies) / len(gm_all_latencies) if gm_all_latencies else 0,
            total_ingestion_time_s=sum(s.ingestion_time_s for s in gm_sample_results),
            avg_entities_per_sample=sum(s.total_entities for s in gm_sample_results) / len(gm_sample_results) if gm_sample_results else 0,
            avg_relationships_per_sample=sum(s.total_relationships for s in gm_sample_results) / len(gm_sample_results) if gm_sample_results else 0,
            samples=gm_sample_results,
        )
        
        naive_results = None
        if compare_naive and naive_sample_results:
            naive_total_q = sum(s.num_questions for s in naive_sample_results)
            naive_total_correct = sum(s.num_correct for s in naive_sample_results)
            naive_all_latencies = [q.latency_ms for s in naive_sample_results for q in s.questions]
            
            naive_results = BenchmarkResults(
                split=split,
                total_samples=len(naive_sample_results),
                total_questions=naive_total_q,
                total_correct=naive_total_correct,
                accuracy=naive_total_correct / naive_total_q if naive_total_q else 0,
                avg_latency_ms=sum(naive_all_latencies) / len(naive_all_latencies) if naive_all_latencies else 0,
                total_ingestion_time_s=sum(s.ingestion_time_s for s in naive_sample_results),
                avg_entities_per_sample=0,
                avg_relationships_per_sample=0,
                samples=naive_sample_results,
            )
        
        # Print comparison
        self._print_results(split, gm_results, naive_results)
        
        return gm_results, naive_results
    
    def _print_results(self, split: str, gm: BenchmarkResults, naive: Optional[BenchmarkResults]):
        """Print comparison results."""
        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {split}")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<30} {'GraphMem':>20}", end="")
        if naive:
            print(f" {'NaiveRAG':>20}")
        else:
            print()
        
        print("-" * 70)
        
        print(f"{'Accuracy':<30} {gm.accuracy:>19.1%}", end="")
        if naive:
            print(f" {naive.accuracy:>19.1%}")
        else:
            print()
        
        print(f"{'Correct / Total':<30} {f'{gm.total_correct}/{gm.total_questions}':>20}", end="")
        if naive:
            print(f" {f'{naive.total_correct}/{naive.total_questions}':>20}")
        else:
            print()
        
        print(f"{'Avg Latency (ms)':<30} {gm.avg_latency_ms:>20.0f}", end="")
        if naive:
            print(f" {naive.avg_latency_ms:>20.0f}")
        else:
            print()
        
        print(f"{'Total Ingestion Time (s)':<30} {gm.total_ingestion_time_s:>20.1f}", end="")
        if naive:
            print(f" {naive.total_ingestion_time_s:>20.1f}")
        else:
            print()
        
        print(f"{'Avg Entities/Sample':<30} {gm.avg_entities_per_sample:>20.0f}")
        print(f"{'Avg Relationships/Sample':<30} {gm.avg_relationships_per_sample:>20.0f}")
        
        # Winner
        print(f"\n{'='*80}")
        if naive:
            if gm.accuracy > naive.accuracy:
                improvement = ((gm.accuracy - naive.accuracy) / naive.accuracy * 100) if naive.accuracy > 0 else 0
                print(f"🏆 GraphMem wins with {improvement:.1f}% higher accuracy!")
            elif naive.accuracy > gm.accuracy:
                improvement = ((naive.accuracy - gm.accuracy) / gm.accuracy * 100) if gm.accuracy > 0 else 0
                print(f"📚 NaiveRAG wins with {improvement:.1f}% higher accuracy")
            else:
                print("🤝 It's a tie!")
        print(f"{'='*80}")
    
    def run_all_splits(
        self,
        max_samples: int = 2,
        max_questions: int = 10,
        compare_naive: bool = True,
    ):
        """Run evaluation on all splits."""
        if self.dataset is None:
            self._load_dataset()
        
        all_gm_results = {}
        all_naive_results = {}
        
        for split in self.SPLITS:
            if split in self.dataset:
                gm, naive = self.run_split(
                    split=split,
                    max_samples=max_samples,
                    max_questions=max_questions,
                    compare_naive=compare_naive,
                )
                all_gm_results[split] = gm
                if naive:
                    all_naive_results[split] = naive
        
        # Overall summary
        self._print_overall_summary(all_gm_results, all_naive_results)
        
        return all_gm_results, all_naive_results
    
    def _print_overall_summary(self, gm_results: Dict, naive_results: Dict):
        """Print overall summary across all splits."""
        print(f"\n{'='*80}")
        print("📊 OVERALL SUMMARY: MemoryAgentBench")
        print(f"{'='*80}")
        
        print(f"\n{'Split':<30} {'GraphMem':>15} {'NaiveRAG':>15} {'Winner':>15}")
        print("-" * 75)
        
        gm_total_correct = 0
        gm_total_questions = 0
        naive_total_correct = 0
        naive_total_questions = 0
        
        for split in self.SPLITS:
            if split in gm_results:
                gm = gm_results[split]
                gm_acc = f"{gm.accuracy:.1%}"
                gm_total_correct += gm.total_correct
                gm_total_questions += gm.total_questions
                
                if split in naive_results:
                    naive = naive_results[split]
                    naive_acc = f"{naive.accuracy:.1%}"
                    naive_total_correct += naive.total_correct
                    naive_total_questions += naive.total_questions
                    
                    if gm.accuracy > naive.accuracy:
                        winner = "GraphMem 🏆"
                    elif naive.accuracy > gm.accuracy:
                        winner = "NaiveRAG"
                    else:
                        winner = "Tie"
                else:
                    naive_acc = "-"
                    winner = "-"
                
                print(f"{split:<30} {gm_acc:>15} {naive_acc:>15} {winner:>15}")
        
        print("-" * 75)
        
        gm_overall = gm_total_correct / gm_total_questions if gm_total_questions else 0
        naive_overall = naive_total_correct / naive_total_questions if naive_total_questions else 0
        
        print(f"{'OVERALL':<30} {gm_overall:>14.1%} {naive_overall:>14.1%}", end="")
        if gm_overall > naive_overall:
            print(f" {'GraphMem 🏆':>15}")
        elif naive_overall > gm_overall:
            print(f" {'NaiveRAG':>15}")
        else:
            print(f" {'Tie':>15}")
        
        print(f"{'='*80}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="MemoryAgentBench Evaluation for GraphMem")
    
    # Provider settings
    parser.add_argument("--provider", default="azure", help="LLM provider")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4.1-mini", help="Azure LLM deployment")
    parser.add_argument("--azure-embedding-deployment", default="text-embedding-3-small", help="Azure embedding deployment")
    
    # Evaluation settings
    parser.add_argument("--split", default=None, help="Specific split to run (or 'all')")
    parser.add_argument("--max-samples", type=int, default=2, help="Max samples per split")
    parser.add_argument("--max-questions", type=int, default=10, help="Max questions per sample")
    parser.add_argument("--no-naive", action="store_true", help="Skip NaiveRAG comparison")
    
    # Options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--turso-db", default="memory_bench.db", help="Turso database path")
    
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
    
    if args.split and args.split != "all":
        evaluator.run_split(
            split=args.split,
            max_samples=args.max_samples,
            max_questions=args.max_questions,
            compare_naive=not args.no_naive,
        )
    else:
        evaluator.run_all_splits(
            max_samples=args.max_samples,
            max_questions=args.max_questions,
            compare_naive=not args.no_naive,
        )


if __name__ == "__main__":
    main()


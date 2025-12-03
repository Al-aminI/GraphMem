#!/usr/bin/env python3
"""
GraphMem vs Naive RAG Evaluation
================================

Using MultiHopRAG dataset from HuggingFace:
- 2556 multi-hop QA samples
- 609 news article corpus

This is an evaluation using a published benchmark dataset.

Author: GraphMem Team
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Add graphmem to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class EvalResult:
    """Single evaluation result."""
    query: str
    expected_answer: str
    graphmem_answer: str
    naive_rag_answer: str
    graphmem_correct: bool
    naive_rag_correct: bool
    question_type: str
    graphmem_time_ms: float = 0
    naive_rag_time_ms: float = 0


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary."""
    name: str
    total_samples: int
    graphmem_correct: int
    naive_rag_correct: int
    graphmem_accuracy: float
    naive_rag_accuracy: float
    graphmem_avg_time_ms: float
    naive_rag_avg_time_ms: float
    by_question_type: Dict[str, Dict] = field(default_factory=dict)


class NaiveRAG:
    """Simple vector-only RAG baseline."""
    
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.docs: List[Dict] = []
        self.embs: List[List[float]] = []
    
    def ingest(self, doc: Dict):
        """Ingest a document."""
        content = f"{doc.get('title', '')}\n{doc.get('body', '')}"
        self.docs.append(doc)
        self.embs.append(self.embeddings.embed_text(content[:8000]))  # Limit length
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query using vector similarity."""
        if not self.docs:
            return "No information available."
        
        q_emb = self.embeddings.embed_text(question)
        
        # Cosine similarity
        scores = []
        for emb in self.embs:
            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
            scores.append(sim)
        
        # Get top-k
        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        # Build context
        context_parts = []
        for i in top_idx:
            doc = self.docs[i]
            context_parts.append(f"Title: {doc.get('title', 'N/A')}\n{doc.get('body', '')[:2000]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # LLM answer
        prompt = f"""Based on the following context, answer the question concisely.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer (be concise):"""
        
        return self.llm.complete(prompt)
    
    def clear(self):
        self.docs = []
        self.embs = []


class MultiHopRAGEvaluator:
    """
    Evaluator using MultiHopRAG dataset.
    
    Compares GraphMem vs Naive RAG on:
    - Multi-hop reasoning
    - Different question types
    - Answer accuracy
    - Query latency
    """
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        llm_model: str = "google/gemini-2.0-flash-001",
        embedding_model: str = "openai/text-embedding-3-small",
        neo4j_uri: str = None,
        neo4j_password: str = None,
        data_dir: str = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_password = neo4j_password
        
        # Data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        
        # Load dataset
        self.corpus = self._load_corpus()
        self.qa_samples = self._load_qa()
        
        self.results: List[EvalResult] = []
    
    def _load_corpus(self) -> List[Dict]:
        """Load corpus documents."""
        corpus_file = os.path.join(self.data_dir, 'multihoprag_corpus.json')
        if not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus not found: {corpus_file}")
        
        with open(corpus_file, 'r') as f:
            return json.load(f)
    
    def _load_qa(self) -> List[Dict]:
        """Load QA samples."""
        qa_file = os.path.join(self.data_dir, 'multihoprag_qa.json')
        if not os.path.exists(qa_file):
            raise FileNotFoundError(f"QA file not found: {qa_file}")
        
        with open(qa_file, 'r') as f:
            return json.load(f)
    
    def _init_naive_rag(self):
        """Initialize Naive RAG baseline."""
        from graphmem.llm.providers import LLMProvider
        from graphmem.llm.embeddings import EmbeddingProvider
        
        llm = LLMProvider(
            provider="openai_compatible",
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
        )
        emb = EmbeddingProvider(
            provider="openai_compatible",
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.embedding_model,
        )
        return NaiveRAG(emb, llm)
    
    def _init_graphmem(self):
        """Initialize GraphMem."""
        from graphmem import GraphMem, MemoryConfig
        
        config = MemoryConfig(
            llm_provider="openai_compatible",
            llm_api_key=self.api_key,
            llm_api_base=self.api_base,
            llm_model=self.llm_model,
            embedding_provider="openai_compatible",
            embedding_api_key=self.api_key,
            embedding_api_base=self.api_base,
            embedding_model=self.embedding_model,
            neo4j_uri=self.neo4j_uri,
            neo4j_username="neo4j",
            neo4j_password=self.neo4j_password,
        )
        return GraphMem(config)
    
    def _check_answer(self, predicted: str, expected: str) -> bool:
        """Check if answer is correct (contains expected answer)."""
        pred_lower = predicted.lower().strip()
        exp_lower = expected.lower().strip()
        
        # Exact match or contains
        if exp_lower in pred_lower:
            return True
        
        # Check key terms
        exp_terms = set(exp_lower.split())
        pred_terms = set(pred_lower.split())
        overlap = exp_terms & pred_terms
        
        # If most expected terms are present, consider correct
        if len(exp_terms) > 0 and len(overlap) / len(exp_terms) >= 0.7:
            return True
        
        return False
    
    def run_evaluation(
        self,
        n_corpus_docs: int = 100,
        n_qa_samples: int = 50,
        seed: int = 42,
    ) -> BenchmarkSummary:
        """
        Run the evaluation.
        
        Args:
            n_corpus_docs: Number of corpus documents to ingest
            n_qa_samples: Number of QA samples to test
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        print("\n" + "=" * 70)
        print("🧪 MULTIHOPRAG EVALUATION: GraphMem vs Naive RAG")
        print("=" * 70)
        print(f"   Dataset: MultiHopRAG (HuggingFace)")
        print(f"   Corpus: {len(self.corpus)} docs (using {n_corpus_docs})")
        print(f"   QA Samples: {len(self.qa_samples)} (testing {n_qa_samples})")
        print(f"   LLM: {self.llm_model}")
        print(f"   Embeddings: {self.embedding_model}")
        print("=" * 70)
        
        # Sample corpus and QA
        corpus_sample = random.sample(self.corpus, min(n_corpus_docs, len(self.corpus)))
        qa_sample = random.sample(self.qa_samples, min(n_qa_samples, len(self.qa_samples)))
        
        # Initialize systems
        print("\n📦 Ingesting corpus into both systems...")
        
        # GraphMem
        print("   → GraphMem...")
        gm = self._init_graphmem()
        gm_ingest_start = time.time()
        for i, doc in enumerate(corpus_sample):
            content = f"{doc.get('title', '')}\n{doc.get('body', '')}"
            try:
                gm.ingest(content[:10000])  # Limit length
            except Exception as e:
                print(f"      Warning: {e}")
            if (i + 1) % 20 == 0:
                print(f"      {i+1}/{len(corpus_sample)} docs ingested")
        gm_ingest_time = time.time() - gm_ingest_start
        print(f"   → GraphMem ingestion: {gm_ingest_time:.1f}s")
        
        # Naive RAG
        print("   → Naive RAG...")
        rag = self._init_naive_rag()
        rag_ingest_start = time.time()
        for doc in corpus_sample:
            try:
                rag.ingest(doc)
            except Exception as e:
                print(f"      Warning: {e}")
        rag_ingest_time = time.time() - rag_ingest_start
        print(f"   → Naive RAG ingestion: {rag_ingest_time:.1f}s")
        
        # Run queries
        print(f"\n🔍 Running {len(qa_sample)} queries...")
        self.results = []
        
        for i, qa in enumerate(qa_sample):
            query = qa['query']
            expected = qa['answer']
            q_type = qa.get('question_type', 'unknown')
            
            # GraphMem query
            try:
                gm_start = time.time()
                gm_resp = gm.query(query)
                gm_time = (time.time() - gm_start) * 1000
                gm_answer = gm_resp.answer
            except Exception as e:
                gm_answer = f"Error: {e}"
                gm_time = 0
            
            # Naive RAG query
            try:
                rag_start = time.time()
                rag_answer = rag.query(query)
                rag_time = (time.time() - rag_start) * 1000
            except Exception as e:
                rag_answer = f"Error: {e}"
                rag_time = 0
            
            # Check correctness
            gm_correct = self._check_answer(gm_answer, expected)
            rag_correct = self._check_answer(rag_answer, expected)
            
            result = EvalResult(
                query=query,
                expected_answer=expected,
                graphmem_answer=gm_answer,
                naive_rag_answer=rag_answer,
                graphmem_correct=gm_correct,
                naive_rag_correct=rag_correct,
                question_type=q_type,
                graphmem_time_ms=gm_time,
                naive_rag_time_ms=rag_time,
            )
            self.results.append(result)
            
            # Progress
            status = "✓" if gm_correct else "✗"
            rag_status = "✓" if rag_correct else "✗"
            print(f"   [{i+1}/{len(qa_sample)}] GM:{status} RAG:{rag_status} | {q_type[:15]:15} | {query[:40]}...")
        
        # Cleanup
        try:
            gm.close()
        except:
            pass
        
        # Calculate summary
        summary = self._calculate_summary()
        
        # Print results
        self._print_results(summary)
        
        return summary
    
    def _calculate_summary(self) -> BenchmarkSummary:
        """Calculate evaluation summary."""
        if not self.results:
            return BenchmarkSummary(
                name="MultiHopRAG",
                total_samples=0,
                graphmem_correct=0,
                naive_rag_correct=0,
                graphmem_accuracy=0,
                naive_rag_accuracy=0,
                graphmem_avg_time_ms=0,
                naive_rag_avg_time_ms=0,
            )
        
        gm_correct = sum(1 for r in self.results if r.graphmem_correct)
        rag_correct = sum(1 for r in self.results if r.naive_rag_correct)
        total = len(self.results)
        
        gm_times = [r.graphmem_time_ms for r in self.results if r.graphmem_time_ms > 0]
        rag_times = [r.naive_rag_time_ms for r in self.results if r.naive_rag_time_ms > 0]
        
        # By question type
        by_type = {}
        for r in self.results:
            q_type = r.question_type
            if q_type not in by_type:
                by_type[q_type] = {"total": 0, "gm_correct": 0, "rag_correct": 0}
            by_type[q_type]["total"] += 1
            if r.graphmem_correct:
                by_type[q_type]["gm_correct"] += 1
            if r.naive_rag_correct:
                by_type[q_type]["rag_correct"] += 1
        
        return BenchmarkSummary(
            name="MultiHopRAG",
            total_samples=total,
            graphmem_correct=gm_correct,
            naive_rag_correct=rag_correct,
            graphmem_accuracy=gm_correct / total if total > 0 else 0,
            naive_rag_accuracy=rag_correct / total if total > 0 else 0,
            graphmem_avg_time_ms=np.mean(gm_times) if gm_times else 0,
            naive_rag_avg_time_ms=np.mean(rag_times) if rag_times else 0,
            by_question_type=by_type,
        )
    
    def _print_results(self, summary: BenchmarkSummary):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("📊 RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"\n   Total samples: {summary.total_samples}")
        print(f"\n   ACCURACY:")
        print(f"   ├── GraphMem:  {summary.graphmem_correct}/{summary.total_samples} = {summary.graphmem_accuracy:.1%}")
        print(f"   └── Naive RAG: {summary.naive_rag_correct}/{summary.total_samples} = {summary.naive_rag_accuracy:.1%}")
        
        print(f"\n   LATENCY (avg):")
        print(f"   ├── GraphMem:  {summary.graphmem_avg_time_ms:.0f}ms")
        print(f"   └── Naive RAG: {summary.naive_rag_avg_time_ms:.0f}ms")
        
        if summary.by_question_type:
            print(f"\n   BY QUESTION TYPE:")
            print(f"   {'Type':<20} {'Total':>6} {'GM':>8} {'RAG':>8}")
            print(f"   {'-'*20} {'-'*6} {'-'*8} {'-'*8}")
            for q_type, stats in summary.by_question_type.items():
                total = stats['total']
                gm_acc = stats['gm_correct'] / total if total > 0 else 0
                rag_acc = stats['rag_correct'] / total if total > 0 else 0
                print(f"   {q_type:<20} {total:>6} {gm_acc:>7.0%} {rag_acc:>7.0%}")
        
        # Winner
        print("\n" + "=" * 70)
        if summary.graphmem_accuracy > summary.naive_rag_accuracy:
            diff = summary.graphmem_accuracy - summary.naive_rag_accuracy
            print(f"   🏆 WINNER: GraphMem (+{diff:.1%} accuracy)")
        elif summary.naive_rag_accuracy > summary.graphmem_accuracy:
            diff = summary.naive_rag_accuracy - summary.graphmem_accuracy
            print(f"   🏆 WINNER: Naive RAG (+{diff:.1%} accuracy)")
        else:
            print(f"   🤝 TIE: Both systems have equal accuracy")
        print("=" * 70)
    
    def save_results(self, filepath: str):
        """Save results to JSON."""
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
            },
            "results": [asdict(r) for r in self.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")


def main():
    """Run the evaluation."""
    # Configuration
    API_KEY = "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"
    API_BASE = "https://openrouter.ai/api/v1"
    LLM_MODEL = "google/gemini-2.0-flash-001"
    EMBEDDING_MODEL = "openai/text-embedding-3-small"
    
    # Optional Neo4j (comment out to use in-memory)
    NEO4J_URI = "neo4j+ssc://ab98af39.databases.neo4j.io"
    NEO4J_PASSWORD = "6n_x28wTO8YOcDUcSOchwtwTNI6vtE7Ns2-sJExYGfQ"
    
    evaluator = MultiHopRAGEvaluator(
        api_key=API_KEY,
        api_base=API_BASE,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        neo4j_uri=NEO4J_URI,
        neo4j_password=NEO4J_PASSWORD,
    )
    
    # Run evaluation with full dataset
    summary = evaluator.run_evaluation(
        n_corpus_docs=609,    # Full corpus (609 docs)
        n_qa_samples=2556,    # Full QA set (2556 samples)
        seed=42,
    )
    
    # Save results
    output_dir = os.path.dirname(__file__)
    evaluator.save_results(os.path.join(output_dir, "eval_results.json"))
    
    return summary


if __name__ == "__main__":
    main()


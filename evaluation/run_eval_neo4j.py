#!/usr/bin/env python3
"""
GraphMem Evaluation with NEO4J + REDIS Backend
===============================================

Enterprise evaluation using Neo4j Graph Database + Redis caching.
Best for production benchmarking with full graph capabilities.

Uses MultiHopRAG dataset from HuggingFace:
- 2556 multi-hop QA samples
- 609 news article corpus

Supports:
- OpenRouter (default)
- Azure OpenAI
- Direct OpenAI
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# Add graphmem to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str, log: bool = True):
        self.name = name
        self.log = log
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.time()
        if self.log:
            logger.info(f"⏱️  START: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.log:
            logger.info(f"✅ DONE:  {self.name} ({self.elapsed:.2f}s)")


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
    backend: str
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
        self.embs.append(self.embeddings.embed_text(content[:8000]))
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query using vector similarity."""
        if not self.docs:
            return "No information available."
        
        q_emb = self.embeddings.embed_text(question)
        
        scores = []
        for emb in self.embs:
            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
            scores.append(sim)
        
        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        context_parts = []
        for i in top_idx:
            doc = self.docs[i]
            context_parts.append(f"Title: {doc.get('title', 'N/A')}\n{doc.get('body', '')[:2000]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
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


class Neo4jEvaluator:
    """
    Evaluator using NEO4J + REDIS backend.
    
    Enterprise configuration with:
    - Neo4j Graph Database (HNSW vector index)
    - Redis caching (query + embedding cache)
    - Full graph capabilities (PageRank, etc.)
    
    Supports providers:
    - openai_compatible (OpenRouter, vLLM, etc.)
    - azure (Azure OpenAI)
    - openai (Direct OpenAI)
    """
    
    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        llm_model: str = "google/gemini-2.0-flash-001",
        embedding_model: str = "openai/text-embedding-3-small",
        neo4j_uri: str = None,
        neo4j_username: str = "neo4j",
        neo4j_password: str = None,
        redis_url: str = None,
        data_dir: str = None,
        # Provider configuration
        llm_provider: str = "openai_compatible",
        embedding_provider: str = "openai_compatible",
        # Azure-specific
        azure_endpoint: str = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.redis_url = redis_url
        
        # Provider config
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment
        self.azure_embedding_deployment = azure_embedding_deployment
        
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        
        logger.info(f"📂 Loading dataset from: {self.data_dir}")
        with Timer("Load corpus"):
            self.corpus = self._load_corpus()
        logger.info(f"   Loaded {len(self.corpus)} corpus documents")
        
        with Timer("Load QA samples"):
            self.qa_samples = self._load_qa()
        logger.info(f"   Loaded {len(self.qa_samples)} QA samples")
        
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
        
        logger.info(f"🤖 Initializing LLM: {self.llm_provider}/{self.llm_model}")
        
        # Build LLM config based on provider
        if self.llm_provider in ("azure", "azure_openai"):
            llm = LLMProvider(
                provider="azure_openai",
                api_key=self.api_key,
                api_base=self.azure_endpoint,
                model=self.azure_deployment or self.llm_model,
                azure_api_version=self.azure_api_version,
            )
        else:
            llm = LLMProvider(
                provider=self.llm_provider,
                api_key=self.api_key,
                api_base=self.api_base,
                model=self.llm_model,
            )
        
        logger.info(f"🔢 Initializing Embeddings: {self.embedding_provider}/{self.embedding_model}")
        
        # Build Embedding config based on provider
        if self.embedding_provider in ("azure", "azure_openai"):
            emb = EmbeddingProvider(
                provider="azure_openai",
                api_key=self.api_key,
                api_base=self.azure_endpoint,
                model=self.azure_embedding_deployment or self.embedding_model,
                azure_api_version=self.azure_api_version,
            )
        else:
            emb = EmbeddingProvider(
                provider=self.embedding_provider,
                api_key=self.api_key,
                api_base=self.api_base,
                model=self.embedding_model,
            )
        
        return NaiveRAG(emb, llm)
    
    def _init_graphmem(self):
        """Initialize GraphMem with NEO4J + REDIS backend."""
        from graphmem import GraphMem, MemoryConfig
        
        logger.info(f"🔧 Building MemoryConfig...")
        logger.info(f"   LLM Provider: {self.llm_provider}")
        logger.info(f"   LLM Model: {self.llm_model}")
        logger.info(f"   Embedding Provider: {self.embedding_provider}")
        logger.info(f"   Embedding Model: {self.embedding_model}")
        logger.info(f"   Neo4j URI: {self.neo4j_uri or 'Not configured (using InMemory)'}")
        logger.info(f"   Redis URL: {'Configured' if self.redis_url else 'Not configured'}")
        
        # Build config based on provider
        if self.llm_provider in ("azure", "azure_openai"):
            config = MemoryConfig(
                llm_provider="azure_openai",
                llm_api_key=self.api_key,
                llm_api_base=self.azure_endpoint,
                llm_model=self.azure_deployment or self.llm_model,
                azure_api_version=self.azure_api_version,
                embedding_provider="azure_openai" if self.embedding_provider in ("azure", "azure_openai") else self.embedding_provider,
                embedding_api_key=self.api_key,
                embedding_api_base=self.azure_endpoint if self.embedding_provider in ("azure", "azure_openai") else self.api_base,
                embedding_model=self.azure_embedding_deployment or self.embedding_model,
                # NEO4J backend
                neo4j_uri=self.neo4j_uri,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password,
                # REDIS caching
                redis_url=self.redis_url,
                redis_ttl=3600,
            )
        else:
            config = MemoryConfig(
                llm_provider=self.llm_provider,
                llm_api_key=self.api_key,
                llm_api_base=self.api_base,
                llm_model=self.llm_model,
                embedding_provider=self.embedding_provider,
                embedding_api_key=self.api_key,
                embedding_api_base=self.api_base,
                embedding_model=self.embedding_model,
                # NEO4J backend
                neo4j_uri=self.neo4j_uri,
                neo4j_username=self.neo4j_username,
                neo4j_password=self.neo4j_password,
                # REDIS caching
                redis_url=self.redis_url,
                redis_ttl=3600,
            )
        
        logger.info(f"🚀 Creating GraphMem instance...")
        return GraphMem(config, user_id="eval_user", memory_id="multihoprag")
    
    def _check_answer(self, predicted: str, expected: str) -> bool:
        """Check if answer is correct."""
        pred_lower = predicted.lower().strip()
        exp_lower = expected.lower().strip()
        
        if exp_lower in pred_lower:
            return True
        
        exp_terms = set(exp_lower.split())
        pred_terms = set(pred_lower.split())
        overlap = exp_terms & pred_terms
        
        if len(exp_terms) > 0 and len(overlap) / len(exp_terms) >= 0.7:
            return True
        
        return False
    
    def run_evaluation(
        self,
        n_corpus_docs: int = 100,
        n_qa_samples: int = 50,
        seed: int = 42,
    ) -> BenchmarkSummary:
        """Run the evaluation with NEO4J + REDIS backend."""
        random.seed(seed)
        
        print("\n" + "=" * 70)
        print("🧪 GRAPHMEM EVALUATION: NEO4J + REDIS Backend")
        print("=" * 70)
        print(f"   Dataset: MultiHopRAG (HuggingFace)")
        print(f"   Corpus: {len(self.corpus)} docs (using {n_corpus_docs})")
        print(f"   QA Samples: {len(self.qa_samples)} (testing {n_qa_samples})")
        print(f"   LLM: {self.llm_model}")
        print(f"   Embeddings: {self.embedding_model}")
        print(f"   Backend: 🏢 NEO4J (Enterprise Graph + HNSW Vector)")
        print(f"   Neo4j: {self.neo4j_uri or 'Not configured'}")
        print(f"   Redis: {'Configured' if self.redis_url else 'Not configured'}")
        print("=" * 70)
        
        if not self.neo4j_uri:
            print("\n⚠️  WARNING: Neo4j URI not provided. Using in-memory fallback.")
        
        corpus_sample = random.sample(self.corpus, min(n_corpus_docs, len(self.corpus)))
        qa_sample = random.sample(self.qa_samples, min(n_qa_samples, len(self.qa_samples)))
        
        print("\n📦 Ingesting corpus into both systems...")
        
        # GraphMem with Neo4j - HIGH-PERFORMANCE BATCH INGESTION
        print("   → GraphMem (Neo4j) - Batch Mode with Auto-Scaling...")
        gm = self._init_graphmem()
        
        # Prepare documents for batch ingestion
        documents = []
        for i, doc in enumerate(corpus_sample):
            content = f"{doc.get('title', '')}\n{doc.get('body', '')}"
            documents.append({
                "id": f"doc_{i}",
                "content": content[:10000],
            })
        
        gm_ingest_start = time.time()
        
        try:
            # High-performance batch ingestion with 10 workers
            # Infinite retry on rate limits - will never fail
            batch_result = gm.ingest_batch(
                documents=documents,
                max_workers=10,    # Hardcoded 10 workers
                show_progress=True,
                auto_scale=False,  # Disable auto-scaling
                aggressive=True,   # Enable retry logic
            )
            gm_ingest_errors = batch_result.get("documents_failed", 0)
            gm_docs_processed = batch_result.get("documents_processed", 0)
            throughput = batch_result.get("throughput_docs_per_sec", 0)
            print(f"      Batch stats: {gm_docs_processed} processed, {gm_ingest_errors} failed, {throughput:.2f} docs/sec")
        except Exception as e:
            print(f"      Batch ingestion failed, falling back to sequential: {e}")
            # Fallback to sequential
            for i, doc in enumerate(documents):
                try:
                    gm.ingest(doc["content"])
                except Exception as e2:
                    print(f"      Warning: {e2}")
                if (i + 1) % 20 == 0:
                    print(f"      {i+1}/{len(documents)} docs ingested")
        
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
            
            status = "✓" if gm_correct else "✗"
            rag_status = "✓" if rag_correct else "✗"
            print(f"   [{i+1}/{len(qa_sample)}] GM:{status} RAG:{rag_status} | {q_type[:15]:15} | {query[:40]}...")
        
        # Cleanup
        try:
            gm.close()
        except:
            pass
        
        summary = self._calculate_summary()
        self._print_results(summary)
        
        return summary
    
    def _calculate_summary(self) -> BenchmarkSummary:
        """Calculate evaluation summary."""
        if not self.results:
            return BenchmarkSummary(
                name="MultiHopRAG",
                backend="Neo4j + Redis",
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
            backend="Neo4j (HNSW Vector) + Redis Cache",
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
        print("📊 RESULTS SUMMARY (NEO4J + REDIS Backend)")
        print("=" * 70)
        
        print(f"\n   Backend: {summary.backend}")
        print(f"   Total samples: {summary.total_samples}")
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
            "backend": "Neo4j + Redis",
            "config": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "neo4j_uri": self.neo4j_uri,
                "redis_configured": bool(self.redis_url),
            },
            "results": [asdict(r) for r in self.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")


def main():
    """Run the NEO4J evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphMem Evaluation with Neo4j + Redis Backend")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""), help="OpenRouter API key")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", ""), help="Neo4j URI")
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""), help="Neo4j password")
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", ""), help="Redis URL")
    parser.add_argument("--corpus-docs", type=int, default=100, help="Number of corpus docs to use")
    parser.add_argument("--qa-samples", type=int, default=50, help="Number of QA samples to test")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (609 docs, 2556 QA)")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your OpenRouter API key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        sys.exit(1)
    
    # Determine sample sizes
    n_corpus = 609 if args.full else args.corpus_docs
    n_qa = 2556 if args.full else args.qa_samples
    
    evaluator = Neo4jEvaluator(
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        llm_model="google/gemini-2.0-flash-001",
        embedding_model="openai/text-embedding-3-small",
        neo4j_uri=args.neo4j_uri or None,
        neo4j_password=args.neo4j_password or None,
        redis_url=args.redis_url or None,
    )
    
    summary = evaluator.run_evaluation(
        n_corpus_docs=n_corpus,
        n_qa_samples=n_qa,
        seed=42,
    )
    
    output_dir = os.path.dirname(__file__)
    evaluator.save_results(os.path.join(output_dir, "eval_results_neo4j.json"))
    
    return summary


if __name__ == "__main__":
    main()


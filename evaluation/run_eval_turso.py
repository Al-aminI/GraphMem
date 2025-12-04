#!/usr/bin/env python3
"""
GraphMem Evaluation with TURSO Backend
=======================================

Lightweight evaluation using Turso (SQLite) - no external servers needed!
Perfect for edge/offline evaluation or when Neo4j/Redis aren't available.

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


class TursoEvaluator:
    """
    Evaluator using TURSO backend (SQLite with native vector search).
    
    No external servers required - perfect for:
    - Edge/offline evaluation
    - CI/CD pipelines
    - Quick benchmarking
    
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
        turso_db_path: str = "eval_memory.db",
        data_dir: str = None,
        # Provider configuration
        llm_provider: str = "openai_compatible",
        embedding_provider: str = "openai_compatible",
        # Azure-specific
        azure_endpoint: str = None,
        azure_api_version: str = "2024-02-15-preview",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.turso_db_path = turso_db_path
        
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
        if self.llm_provider == "azure":
            llm = LLMProvider(
                provider="azure",
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
        if self.embedding_provider == "azure":
            emb = EmbeddingProvider(
                provider="azure",
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
        """Initialize GraphMem with TURSO backend."""
        from graphmem import GraphMem, MemoryConfig
        
        # Clean up old database
        if os.path.exists(self.turso_db_path):
            logger.info(f"🗑️  Removing old database: {self.turso_db_path}")
            os.remove(self.turso_db_path)
        
        logger.info(f"🔧 Building MemoryConfig...")
        logger.info(f"   LLM Provider: {self.llm_provider}")
        logger.info(f"   LLM Model: {self.llm_model}")
        logger.info(f"   Embedding Provider: {self.embedding_provider}")
        logger.info(f"   Embedding Model: {self.embedding_model}")
        logger.info(f"   Turso DB: {self.turso_db_path}")
        
        # Build config based on provider
        if self.llm_provider == "azure":
            config = MemoryConfig(
                llm_provider="azure",
                llm_api_key=self.api_key,
                llm_api_base=self.azure_endpoint,
                llm_model=self.azure_deployment or self.llm_model,
                azure_api_version=self.azure_api_version,
                embedding_provider="azure" if self.embedding_provider == "azure" else self.embedding_provider,
                embedding_api_key=self.api_key,
                embedding_api_base=self.azure_endpoint if self.embedding_provider == "azure" else self.api_base,
                embedding_model=self.azure_embedding_deployment or self.embedding_model,
                turso_db_path=self.turso_db_path,
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
                turso_db_path=self.turso_db_path,
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
        """Run the evaluation with TURSO backend."""
        random.seed(seed)
        
        eval_start = time.time()
        
        print("\n" + "=" * 70)
        print("🧪 GRAPHMEM EVALUATION: TURSO Backend")
        print("=" * 70)
        print(f"   Dataset: MultiHopRAG (HuggingFace)")
        print(f"   Corpus: {len(self.corpus)} docs (using {n_corpus_docs})")
        print(f"   QA Samples: {len(self.qa_samples)} (testing {n_qa_samples})")
        print(f"   LLM Provider: {self.llm_provider}")
        print(f"   LLM Model: {self.llm_model}")
        print(f"   Embedding Provider: {self.embedding_provider}")
        print(f"   Embeddings: {self.embedding_model}")
        print(f"   Backend: 🔥 TURSO (SQLite + native vector search)")
        print(f"   Database: {self.turso_db_path}")
        if self.llm_provider == "azure":
            print(f"   Azure Endpoint: {self.azure_endpoint}")
        print("=" * 70)
        
        logger.info("📊 PHASE 1: Sampling data")
        corpus_sample = random.sample(self.corpus, min(n_corpus_docs, len(self.corpus)))
        qa_sample = random.sample(self.qa_samples, min(n_qa_samples, len(self.qa_samples)))
        logger.info(f"   Selected {len(corpus_sample)} docs and {len(qa_sample)} QA samples")
        
        # ==================== PHASE 2: Initialize Systems ====================
        logger.info("📊 PHASE 2: Initializing systems")
        
        with Timer("Initialize GraphMem (Turso)"):
            gm = self._init_graphmem()
        
        with Timer("Initialize Naive RAG"):
            rag = self._init_naive_rag()
        
        # ==================== PHASE 3: Ingest Corpus ====================
        logger.info("📊 PHASE 3: Ingesting corpus")
        
        # GraphMem ingestion with detailed logging
        logger.info("📦 Ingesting into GraphMem...")
        gm_ingest_start = time.time()
        gm_ingest_errors = 0
        
        for i, doc in enumerate(corpus_sample):
            doc_start = time.time()
            content = f"{doc.get('title', '')}\n{doc.get('body', '')}"
            try:
                gm.ingest(content[:10000])
                doc_time = time.time() - doc_start
                if (i + 1) % 10 == 0:
                    logger.info(f"   GraphMem: {i+1}/{len(corpus_sample)} docs ({doc_time:.2f}s/doc)")
            except Exception as e:
                gm_ingest_errors += 1
                logger.warning(f"   ⚠️ GraphMem ingest error [{i+1}]: {str(e)[:100]}")
                if gm_ingest_errors >= 5:
                    logger.error(f"   ❌ Too many errors ({gm_ingest_errors}), check API key/endpoint!")
        
        gm_ingest_time = time.time() - gm_ingest_start
        logger.info(f"✅ GraphMem ingestion complete: {gm_ingest_time:.1f}s ({gm_ingest_errors} errors)")
        
        # Save memory to Turso
        with Timer("Save memory to Turso"):
            try:
                gm.save()
                logger.info(f"   Memory saved to: {self.turso_db_path}")
            except Exception as e:
                logger.warning(f"   Could not save memory: {e}")
        
        # Naive RAG ingestion
        logger.info("📦 Ingesting into Naive RAG...")
        rag_ingest_start = time.time()
        rag_ingest_errors = 0
        
        for i, doc in enumerate(corpus_sample):
            try:
                rag.ingest(doc)
                if (i + 1) % 50 == 0:
                    logger.info(f"   Naive RAG: {i+1}/{len(corpus_sample)} docs")
            except Exception as e:
                rag_ingest_errors += 1
                logger.warning(f"   ⚠️ RAG ingest error [{i+1}]: {str(e)[:100]}")
        
        rag_ingest_time = time.time() - rag_ingest_start
        logger.info(f"✅ Naive RAG ingestion complete: {rag_ingest_time:.1f}s ({rag_ingest_errors} errors)")
        
        # ==================== PHASE 4: Run Queries ====================
        logger.info(f"📊 PHASE 4: Running {len(qa_sample)} queries")
        self.results = []
        
        gm_total_time = 0
        rag_total_time = 0
        gm_errors = 0
        rag_errors = 0
        
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
                gm_total_time += gm_time
            except Exception as e:
                gm_answer = f"Error: {e}"
                gm_time = 0
                gm_errors += 1
                logger.warning(f"   ⚠️ GraphMem query error [{i+1}]: {str(e)[:80]}")
            
            # Naive RAG query
            try:
                rag_start = time.time()
                rag_answer = rag.query(query)
                rag_time = (time.time() - rag_start) * 1000
                rag_total_time += rag_time
            except Exception as e:
                rag_answer = f"Error: {e}"
                rag_time = 0
                rag_errors += 1
                logger.warning(f"   ⚠️ RAG query error [{i+1}]: {str(e)[:80]}")
            
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
            
            # Detailed progress every 10 queries
            if (i + 1) % 10 == 0:
                gm_avg = gm_total_time / (i + 1)
                rag_avg = rag_total_time / (i + 1)
                logger.info(f"   Query {i+1}/{len(qa_sample)} | GM:{gm_avg:.0f}ms RAG:{rag_avg:.0f}ms | Errors: GM={gm_errors} RAG={rag_errors}")
            
            # Always print individual result
            print(f"   [{i+1}/{len(qa_sample)}] GM:{status} RAG:{rag_status} | {q_type[:15]:15} | {query[:40]}...")
        
        # ==================== PHASE 5: Cleanup ====================
        logger.info("📊 PHASE 5: Cleanup")
        with Timer("Close GraphMem"):
            try:
                gm.close()
            except:
                pass
        
        # ==================== PHASE 6: Calculate Results ====================
        logger.info("📊 PHASE 6: Calculating results")
        
        summary = self._calculate_summary()
        
        total_time = time.time() - eval_start
        logger.info(f"🎉 EVALUATION COMPLETE in {total_time:.1f}s")
        logger.info(f"   Total errors: GraphMem={gm_errors + gm_ingest_errors}, RAG={rag_errors + rag_ingest_errors}")
        
        self._print_results(summary)
        
        return summary
    
    def _calculate_summary(self) -> BenchmarkSummary:
        """Calculate evaluation summary."""
        if not self.results:
            return BenchmarkSummary(
                name="MultiHopRAG",
                backend="Turso",
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
            backend="Turso (SQLite + Native Vector)",
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
        print("📊 RESULTS SUMMARY (TURSO Backend)")
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
            "backend": "Turso",
            "config": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "turso_db_path": self.turso_db_path,
            },
            "results": [asdict(r) for r in self.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")


def main():
    """Run the TURSO evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GraphMem Evaluation with Turso Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # OpenRouter (default)
  python run_eval_turso.py --api-key YOUR_OPENROUTER_KEY --full
  
  # Azure OpenAI
  python run_eval_turso.py --provider azure \\
    --api-key YOUR_AZURE_KEY \\
    --azure-endpoint https://YOUR_RESOURCE.openai.azure.com \\
    --azure-deployment gpt-4.1 \\
    --azure-embedding-deployment text-embedding-ada-002 \\
    --full
  
  # Direct OpenAI
  python run_eval_turso.py --provider openai --api-key YOUR_OPENAI_KEY --full
        """
    )
    
    # Provider selection
    parser.add_argument("--provider", choices=["openrouter", "azure", "openai"], default="openrouter",
                       help="LLM provider (default: openrouter)")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="API key")
    
    # OpenRouter/OpenAI settings
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument("--llm-model", default="google/gemini-2.0-flash-001", help="LLM model name")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small", help="Embedding model")
    
    # Azure settings
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", help="Azure LLM deployment name")
    parser.add_argument("--azure-embedding-deployment", help="Azure embedding deployment name")
    parser.add_argument("--azure-api-version", default="2024-02-15-preview", help="Azure API version")
    
    # Evaluation settings
    parser.add_argument("--corpus-docs", type=int, default=100, help="Number of corpus docs")
    parser.add_argument("--qa-samples", type=int, default=50, help="Number of QA samples")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (609 docs, 2556 QA)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        # Try environment variables
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        api_key = input(f"Enter your {args.provider.upper()} API key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        sys.exit(1)
    
    # Determine sample sizes
    n_corpus = 609 if args.full else args.corpus_docs
    n_qa = 2556 if args.full else args.qa_samples
    
    # Configure based on provider
    if args.provider == "azure":
        if not args.azure_endpoint:
            args.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or input("Enter Azure endpoint: ").strip()
        if not args.azure_deployment:
            args.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "gpt-4.1"
        if not args.azure_embedding_deployment:
            args.azure_embedding_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT") or "text-embedding-ada-002"
        
        logger.info(f"🔷 Using Azure OpenAI")
        logger.info(f"   Endpoint: {args.azure_endpoint}")
        logger.info(f"   LLM Deployment: {args.azure_deployment}")
        logger.info(f"   Embedding Deployment: {args.azure_embedding_deployment}")
        
        evaluator = TursoEvaluator(
            api_key=api_key,
            llm_provider="azure",
            embedding_provider="azure",
            azure_endpoint=args.azure_endpoint,
            azure_deployment=args.azure_deployment,
            azure_embedding_deployment=args.azure_embedding_deployment,
            azure_api_version=args.azure_api_version,
            llm_model=args.azure_deployment,
            embedding_model=args.azure_embedding_deployment,
            turso_db_path="eval_turso_memory.db",
        )
    
    elif args.provider == "openai":
        logger.info(f"🟢 Using Direct OpenAI")
        evaluator = TursoEvaluator(
            api_key=api_key,
            api_base="https://api.openai.com/v1",
            llm_provider="openai",
            embedding_provider="openai",
            llm_model=args.llm_model if args.llm_model != "google/gemini-2.0-flash-001" else "gpt-4-turbo-preview",
            embedding_model=args.embedding_model if args.embedding_model != "openai/text-embedding-3-small" else "text-embedding-3-small",
            turso_db_path="eval_turso_memory.db",
        )
    
    else:  # openrouter (default)
        logger.info(f"🌐 Using OpenRouter")
        evaluator = TursoEvaluator(
            api_key=api_key,
            api_base=args.api_base,
            llm_provider="openai_compatible",
            embedding_provider="openai_compatible",
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            turso_db_path="eval_turso_memory.db",
        )
    
    # Run evaluation
    summary = evaluator.run_evaluation(
        n_corpus_docs=n_corpus,
        n_qa_samples=n_qa,
        seed=42,
    )
    
    # Save results
    output_dir = os.path.dirname(__file__)
    evaluator.save_results(os.path.join(output_dir, f"eval_results_turso_{args.provider}.json"))
    
    return summary


if __name__ == "__main__":
    main()


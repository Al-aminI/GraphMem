#!/usr/bin/env python3
"""
Comprehensive GraphMem Evaluation with RAGAS-Style Metrics
============================================================

Implements RAGAS (Retrieval-Augmented Generation Assessment) methodology:
https://docs.ragas.io/en/stable/

RAGAS Metrics:
1. FAITHFULNESS: Is the answer grounded in the retrieved context?
2. ANSWER RELEVANCY: Does the answer address the question?
3. CONTEXT PRECISION: Are the retrieved documents relevant?
4. CONTEXT RECALL: Does the context contain info needed for the answer?

Also includes:
- Token efficiency comparison
- Cost tracking
- Latency breakdown
- Memory metrics

Usage:
    python comprehensive_eval.py \
        --provider azure \
        --api-key "YOUR_KEY" \
        --azure-endpoint "https://xxx.openai.azure.com/" \
        --azure-deployment "gpt-4.1-mini" \
        --azure-embedding-deployment "text-embedding-ada-002" \
        --corpus-docs 100 \
        --qa-samples 50 \
        --debug
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphmem import GraphMem, MemoryConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PRICING (per 1K tokens)
# ============================================================================
PRICING = {
    "gpt-4.1-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0},
    "text-embedding-3-small": {"input": 0.00002, "output": 0},
}


# ============================================================================
# RAGAS-STYLE PROMPTS (LLM-as-Judge)
# ============================================================================
FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an answer to the retrieved context.

Faithfulness measures whether EVERY claim in the answer can be verified from the context.
A faithful answer only contains information that is supported by the context.

Context:
{context}

Question: {question}

Answer: {answer}

Instructions:
1. Identify all factual claims made in the answer
2. For each claim, check if it can be verified from the context
3. Score from 0.0 to 1.0 where:
   - 1.0 = All claims are supported by context
   - 0.5 = Some claims supported, some not
   - 0.0 = No claims supported or answer contradicts context

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reason": "<brief explanation>"}}"""

ANSWER_RELEVANCY_PROMPT = """You are evaluating how relevant an answer is to the question asked.

Answer Relevancy measures whether the answer directly and completely addresses the question.
A relevant answer should be focused, complete, and not contain unnecessary information.

Question: {question}

Answer: {answer}

Instructions:
1. Does the answer directly address what was asked?
2. Is the answer complete or does it miss key information?
3. Does the answer contain irrelevant or redundant information?
4. Score from 0.0 to 1.0 where:
   - 1.0 = Answer perfectly addresses the question
   - 0.5 = Answer partially addresses the question
   - 0.0 = Answer is irrelevant or doesn't address the question

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reason": "<brief explanation>"}}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of retrieved context for answering a question.

Context Precision measures whether the retrieved information is relevant and useful.
High precision means the context contains information that helps answer the question.

Question: {question}

Retrieved Context:
{context}

Instructions:
1. Is the context relevant to the question?
2. Does it contain information that could help answer the question?
3. Is there a lot of irrelevant noise in the context?
4. Score from 0.0 to 1.0 where:
   - 1.0 = Context is highly relevant and focused
   - 0.5 = Context is somewhat relevant with some noise
   - 0.0 = Context is completely irrelevant

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reason": "<brief explanation>"}}"""

CONTEXT_RECALL_PROMPT = """You are evaluating whether the retrieved context contains all information needed to answer the question correctly.

Context Recall measures if the retrieval system found all the necessary evidence.
High recall means all the information needed for the correct answer is in the context.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context}

Instructions:
1. Does the context contain the information needed to derive the ground truth answer?
2. Is there sufficient evidence in the context to support the correct answer?
3. Score from 0.0 to 1.0 where:
   - 1.0 = Context contains all information needed for the answer
   - 0.5 = Context contains some but not all needed information
   - 0.0 = Context is missing critical information

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reason": "<brief explanation>"}}"""

ANSWER_CORRECTNESS_PROMPT = """You are evaluating whether an answer is correct compared to the ground truth.

Answer Correctness measures factual accuracy and semantic similarity to the expected answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {answer}

Instructions:
1. Does the generated answer convey the same meaning as the ground truth?
2. Are the facts in the generated answer correct?
3. Consider that answers may be phrased differently but still be correct.
4. Score from 0.0 to 1.0 where:
   - 1.0 = Answer is correct and matches ground truth semantically
   - 0.5 = Answer is partially correct
   - 0.0 = Answer is wrong or contradicts ground truth

Respond with ONLY a JSON object:
{{"score": <float 0-1>, "reason": "<brief explanation>"}}"""


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class RAGASMetrics:
    """RAGAS-style evaluation metrics (0-1 scores)."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    
    # Reasons from LLM judge
    faithfulness_reason: str = ""
    answer_relevancy_reason: str = ""
    context_precision_reason: str = ""
    context_recall_reason: str = ""
    answer_correctness_reason: str = ""
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            self.faithfulness * 0.2 +
            self.answer_relevancy * 0.2 +
            self.context_precision * 0.2 +
            self.context_recall * 0.2 +
            self.answer_correctness * 0.2
        )


@dataclass
class TokenMetrics:
    """Token usage metrics."""
    context_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0


@dataclass
class CostMetrics:
    """Cost metrics in USD."""
    llm_input_cost: float = 0.0
    llm_output_cost: float = 0.0
    embedding_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class LatencyMetrics:
    """Latency breakdown in ms."""
    retrieval_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class MemoryMetrics:
    """Memory efficiency metrics."""
    total_documents: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    total_clusters: int = 0
    unique_entity_names: int = 0
    deduplication_ratio: float = 0.0
    entities_per_doc: float = 0.0
    relationships_per_entity: float = 0.0
    storage_bytes: int = 0


@dataclass
class AccuracyByType:
    """Accuracy by question type (from MultiHop-RAG paper)."""
    inference_correct: int = 0
    inference_total: int = 0
    comparison_correct: int = 0
    comparison_total: int = 0
    temporal_correct: int = 0
    temporal_total: int = 0
    null_correct: int = 0
    null_total: int = 0


@dataclass
class QueryResult:
    """Single query result with full details including RAGAS metrics."""
    query: str
    expected: str
    predicted: str
    context: str  # The retrieved context!
    correct: bool  # Legacy simple matching
    question_type: str
    ragas: RAGASMetrics = field(default_factory=RAGASMetrics)
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)


@dataclass
class SystemResults:
    """Results for one system (GraphMem or NaiveRAG)."""
    system_name: str = ""
    
    # Summary
    total_queries: int = 0
    correct_queries: int = 0
    accuracy: float = 0.0
    
    # RAGAS Averages
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_correctness: float = 0.0
    avg_ragas_score: float = 0.0
    
    # By type
    accuracy_breakdown: AccuracyByType = field(default_factory=AccuracyByType)
    
    # Efficiency
    avg_context_tokens: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_query: float = 0.0
    
    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # Memory
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    
    # Individual results
    queries: List[QueryResult] = field(default_factory=list)


# ============================================================================
# NAIVE RAG BASELINE
# ============================================================================
class NaiveRAG:
    """
    Simple RAG baseline (like the paper uses):
    - Chunk documents
    - Embed chunks
    - Retrieve top-K by cosine similarity
    - Generate answer with LLM
    """
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: str = None,
        api_base: str = None,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-ada-002",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
        chunk_size: int = 512,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.azure_deployment = azure_deployment
        self.azure_embedding_deployment = azure_embedding_deployment
        self.chunk_size = chunk_size
        
        # Storage
        self.chunks: List[Dict] = []
        self.embeddings: List[List[float]] = []
        
        # Initialize providers
        self._init_providers()
    
    def _init_providers(self):
        """Initialize LLM and embedding providers."""
        from graphmem.llm.providers import get_llm_provider
        from graphmem.llm.embeddings import get_embedding_provider
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        self.llm = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,  # Fixed: use 'deployment' not 'azure_deployment'
        )
        
        self.embedder = get_embedding_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.embedding_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_embedding_deployment,  # Fixed: use 'deployment' not 'azure_deployment'
        )
    
    def ingest(self, content: str, doc_id: str):
        """Chunk and embed document."""
        words = content.split()
        chunk_words = []
        chunk_idx = 0
        
        for word in words:
            chunk_words.append(word)
            if len(' '.join(chunk_words)) >= self.chunk_size:
                chunk_content = ' '.join(chunk_words)
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                
                try:
                    # Use embed_text (correct method name)
                    embedding = self.embedder.embed_text(chunk_content)
                    self.chunks.append({
                        "id": chunk_id,
                        "content": chunk_content,
                        "doc_id": doc_id,
                    })
                    self.embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
                
                chunk_words = []
                chunk_idx += 1
        
        # Remaining
        if chunk_words:
            chunk_content = ' '.join(chunk_words)
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            try:
                # Use embed_text (correct method name)
                embedding = self.embedder.embed_text(chunk_content)
                self.chunks.append({
                    "id": chunk_id,
                    "content": chunk_content,
                    "doc_id": doc_id,
                })
                self.embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def retrieve(self, query: str, top_k: int = 6) -> Tuple[List[Dict], str]:
        """Retrieve top-K chunks by similarity. Returns chunks and combined context."""
        if not self.embeddings:
            return [], ""
        
        query_embedding = self.embedder.embed_text(query)
        
        similarities = []
        for i, chunk_emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, chunk_emb)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, sim in similarities[:top_k]:
            results.append({
                **self.chunks[i],
                "similarity": sim,
            })
        
        context = "\n\n---\n\n".join([r["content"] for r in results])
        return results, context
    
    def query(self, query: str, top_k: int = 6) -> Tuple[str, str, TokenMetrics, LatencyMetrics]:
        """Query and return (answer, context, tokens, latency)."""
        tokens = TokenMetrics()
        latency = LatencyMetrics()
        
        # Retrieval
        retrieval_start = time.perf_counter()
        retrieved, context = self.retrieve(query, top_k)
        retrieval_end = time.perf_counter()
        latency.retrieval_ms = (retrieval_end - retrieval_start) * 1000
        
        tokens.context_tokens = len(context) // 4
        
        # Generate
        llm_start = time.perf_counter()
        
        prompt = f"""Answer the question based ONLY on the provided context.
If the answer cannot be determined from the context, say "Insufficient information".
Give a short, direct answer (single word or phrase when possible).

Context:
{context}

Question: {query}

Answer:"""
        
        tokens.prompt_tokens = len(prompt) // 4
        
        try:
            # Use proper message format for chat API
            messages = [{"role": "user", "content": prompt}]
            answer = self.llm.chat(messages)
            tokens.completion_tokens = len(answer) // 4
        except Exception as e:
            answer = f"Error: {e}"
            tokens.completion_tokens = 10
        
        llm_end = time.perf_counter()
        latency.llm_generation_ms = (llm_end - llm_start) * 1000
        latency.total_ms = latency.retrieval_ms + latency.llm_generation_ms
        tokens.total_tokens = tokens.prompt_tokens + tokens.completion_tokens
        
        return answer, context, tokens, latency


# ============================================================================
# RAGAS EVALUATOR (LLM-as-Judge)
# ============================================================================
class RAGASEvaluator:
    """
    RAGAS-style evaluation using LLM-as-judge.
    
    Evaluates:
    - Faithfulness: Answer grounded in context?
    - Answer Relevancy: Answer addresses question?
    - Context Precision: Retrieved context relevant?
    - Context Recall: Context has needed info?
    - Answer Correctness: Answer matches ground truth?
    """
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"score": 0.0, "reason": "Failed to parse response"}
        except Exception as e:
            return {"score": 0.0, "reason": f"Parse error: {e}"}
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with proper message format."""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.chat(messages)
    
    def evaluate_faithfulness(self, question: str, context: str, answer: str) -> Tuple[float, str]:
        """Evaluate if answer is grounded in context."""
        if not context or not answer:
            return 0.0, "Missing context or answer"
        
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:4000],  # Limit context
            question=question,
            answer=answer
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> Tuple[float, str]:
        """Evaluate if answer addresses the question."""
        if not answer:
            return 0.0, "Missing answer"
        
        prompt = ANSWER_RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def evaluate_context_precision(self, question: str, context: str) -> Tuple[float, str]:
        """Evaluate if retrieved context is relevant."""
        if not context:
            return 0.0, "Missing context"
        
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context=context[:4000]
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def evaluate_context_recall(self, question: str, context: str, ground_truth: str) -> Tuple[float, str]:
        """Evaluate if context contains info needed for the answer."""
        if not context:
            return 0.0, "Missing context"
        
        prompt = CONTEXT_RECALL_PROMPT.format(
            question=question,
            context=context[:4000],
            ground_truth=ground_truth
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def evaluate_answer_correctness(self, question: str, answer: str, ground_truth: str) -> Tuple[float, str]:
        """Evaluate if answer matches ground truth."""
        if not answer:
            return 0.0, "Missing answer"
        
        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            return float(result.get("score", 0)), result.get("reason", "")
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def evaluate_all(
        self, 
        question: str, 
        context: str, 
        answer: str, 
        ground_truth: str
    ) -> RAGASMetrics:
        """Run all RAGAS evaluations."""
        metrics = RAGASMetrics()
        
        # Faithfulness
        metrics.faithfulness, metrics.faithfulness_reason = self.evaluate_faithfulness(
            question, context, answer
        )
        
        # Answer Relevancy
        metrics.answer_relevancy, metrics.answer_relevancy_reason = self.evaluate_answer_relevancy(
            question, answer
        )
        
        # Context Precision
        metrics.context_precision, metrics.context_precision_reason = self.evaluate_context_precision(
            question, context
        )
        
        # Context Recall
        metrics.context_recall, metrics.context_recall_reason = self.evaluate_context_recall(
            question, context, ground_truth
        )
        
        # Answer Correctness
        metrics.answer_correctness, metrics.answer_correctness_reason = self.evaluate_answer_correctness(
            question, answer, ground_truth
        )
        
        return metrics


# ============================================================================
# COMPREHENSIVE EVALUATOR
# ============================================================================
class ComprehensiveEvaluator:
    """
    Comprehensive evaluation comparing GraphMem vs Naive RAG with RAGAS metrics.
    """
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: str = None,
        api_base: str = None,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-ada-002",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
        turso_db_path: str = "eval_graphmem.db",
        debug: bool = False,
        skip_ingestion: bool = False,
        use_ragas: bool = True,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.azure_deployment = azure_deployment
        self.azure_embedding_deployment = azure_embedding_deployment
        self.turso_db_path = turso_db_path
        self.debug = debug
        self.skip_ingestion = skip_ingestion
        self.use_ragas = use_ragas
        
        # Load data
        self.data_dir = Path(__file__).parent / "data"
        self.corpus = self._load_corpus()
        self.qa_samples = self._load_qa()
        
        # Results
        self.graphmem_results = SystemResults(system_name="GraphMem")
        self.naive_results = SystemResults(system_name="NaiveRAG")
        
        # Debug log
        self.debug_log: List[Dict] = []
        
        # RAGAS evaluator (initialized later)
        self.ragas_evaluator: Optional[RAGASEvaluator] = None
    
    def _load_corpus(self) -> List[Dict]:
        """Load corpus documents."""
        possible_files = [
            self.data_dir / "multihoprag_corpus.json",
            self.data_dir / "corpus.json",
        ]
        
        for corpus_file in possible_files:
            if corpus_file.exists():
                logger.info(f"Loading corpus from: {corpus_file}")
                with open(corpus_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} corpus documents")
                return data
        
        logger.error(f"No corpus file found in {self.data_dir}")
        return []
    
    def _load_qa(self) -> List[Dict]:
        """Load QA samples."""
        possible_files = [
            self.data_dir / "multihoprag_qa.json",
            self.data_dir / "qa_samples.json",
            self.data_dir / "qa.json",
        ]
        
        for qa_file in possible_files:
            if qa_file.exists():
                logger.info(f"Loading QA from: {qa_file}")
                with open(qa_file) as f:
                    data = json.load(f)
                logger.info(f"Loaded {len(data)} QA samples")
                return data
        
        logger.error(f"No QA file found in {self.data_dir}")
        return []
    
    def _find_aligned_corpus(self, qa_sample: Dict) -> List[Dict]:
        """
        Find corpus documents that match the evidence in a QA sample.
        
        Uses title and source matching to find relevant documents.
        """
        evidence_list = qa_sample.get("evidence_list", [])
        if not evidence_list:
            return []
        
        # Extract evidence titles and sources
        evidence_markers = []
        for ev in evidence_list:
            title = ev.get("title", "").lower()
            source = ev.get("source", "").lower()
            evidence_markers.append((title, source))
        
        # Find matching corpus docs
        matched_docs = []
        for doc in self.corpus:
            doc_title = doc.get("title", "").lower()
            doc_source = doc.get("source", "").lower()
            
            for ev_title, ev_source in evidence_markers:
                # Match by title similarity or source
                if ev_title and ev_title[:50] in doc_title:
                    matched_docs.append(doc)
                    break
                elif ev_source and ev_source in doc_source and ev_title[:30] in doc_title:
                    matched_docs.append(doc)
                    break
        
        return matched_docs
    
    def _init_graphmem(self) -> GraphMem:
        """Initialize GraphMem with Turso backend."""
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
        
        return GraphMem(config)
    
    def _init_naive_rag(self) -> NaiveRAG:
        """Initialize Naive RAG baseline."""
        return NaiveRAG(
            provider=self.provider,
            api_key=self.api_key,
            api_base=self.api_base,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            azure_deployment=self.azure_deployment,
            azure_embedding_deployment=self.azure_embedding_deployment,
        )
    
    def _init_ragas_evaluator(self):
        """Initialize RAGAS evaluator with LLM."""
        from graphmem.llm.providers import get_llm_provider
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        llm = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,  # Fixed: use 'deployment' not 'azure_deployment'
        )
        
        self.ragas_evaluator = RAGASEvaluator(llm)
    
    def _get_question_type(self, qa: Dict) -> str:
        """Get question type from QA sample."""
        q_type = qa.get("question_type", "").lower()
        
        if "inference" in q_type:
            return "inference"
        elif "comparison" in q_type:
            return "comparison"
        elif "temporal" in q_type:
            return "temporal"
        elif "null" in q_type:
            return "null"
        else:
            query = qa.get("query", "").lower()
            if "before" in query or "after" in query or "when" in query:
                return "temporal"
            elif "compare" in query or "both" in query or "same" in query:
                return "comparison"
            else:
                return "inference"
    
    def _simple_check_answer(self, predicted: str, expected: str) -> bool:
        """Simple string matching (legacy)."""
        pred = predicted.lower().strip()
        exp = expected.lower().strip()
        
        if exp in pred:
            return True
        
        if exp == "yes" and any(x in pred for x in ["yes", "correct", "true"]):
            return True
        if exp == "no" and "no" in pred and "don't know" not in pred:
            return True
        
        exp_terms = set(exp.split())
        pred_terms = set(pred.split())
        if exp_terms:
            overlap = len(exp_terms & pred_terms) / len(exp_terms)
            if overlap >= 0.6:
                return True
        
        return False
    
    def _calculate_cost(self, tokens: TokenMetrics, model: str) -> CostMetrics:
        """Calculate cost in USD."""
        cost = CostMetrics()
        pricing = PRICING.get(model, {"input": 0.001, "output": 0.002})
        cost.llm_input_cost = (tokens.prompt_tokens / 1000) * pricing["input"]
        cost.llm_output_cost = (tokens.completion_tokens / 1000) * pricing["output"]
        cost.total_cost = cost.llm_input_cost + cost.llm_output_cost
        return cost
    
    def _update_accuracy_breakdown(self, results: SystemResults, q_type: str, correct: bool):
        """Update accuracy breakdown by question type."""
        ab = results.accuracy_breakdown
        
        if q_type == "inference":
            ab.inference_total += 1
            if correct:
                ab.inference_correct += 1
        elif q_type == "comparison":
            ab.comparison_total += 1
            if correct:
                ab.comparison_correct += 1
        elif q_type == "temporal":
            ab.temporal_total += 1
            if correct:
                ab.temporal_correct += 1
        elif q_type == "null":
            ab.null_total += 1
            if correct:
                ab.null_correct += 1
    
    def _log_debug(
        self, 
        system: str, 
        query: str, 
        expected: str, 
        predicted: str, 
        context: str,
        ragas: RAGASMetrics,
        q_type: str
    ):
        """Log debug information with RAGAS scores."""
        entry = {
            "system": system,
            "question_type": q_type,
            "query": query[:200],
            "expected": expected,
            "predicted": predicted[:500],
            "context_preview": context[:500] if context else "",
            "ragas_scores": {
                "faithfulness": ragas.faithfulness,
                "answer_relevancy": ragas.answer_relevancy,
                "context_precision": ragas.context_precision,
                "context_recall": ragas.context_recall,
                "answer_correctness": ragas.answer_correctness,
                "overall": ragas.overall_score,
            },
            "ragas_reasons": {
                "faithfulness": ragas.faithfulness_reason,
                "answer_relevancy": ragas.answer_relevancy_reason,
                "context_precision": ragas.context_precision_reason,
                "context_recall": ragas.context_recall_reason,
                "answer_correctness": ragas.answer_correctness_reason,
            }
        }
        self.debug_log.append(entry)
        
        if self.debug:
            status = "✅" if ragas.answer_correctness >= 0.5 else "❌"
            logger.info(f"\n{'='*70}")
            logger.info(f"[{system}] {status} {q_type.upper()} | RAGAS: {ragas.overall_score:.2f}")
            logger.info(f"Q: {query[:100]}...")
            logger.info(f"Expected: {expected}")
            logger.info(f"Predicted: {predicted[:150]}...")
            logger.info(f"Context: {context[:200]}..." if context else "Context: (none)")
            logger.info(f"RAGAS Scores:")
            logger.info(f"  Faithfulness: {ragas.faithfulness:.2f} - {ragas.faithfulness_reason[:80]}")
            logger.info(f"  Relevancy:    {ragas.answer_relevancy:.2f} - {ragas.answer_relevancy_reason[:80]}")
            logger.info(f"  Ctx Precision:{ragas.context_precision:.2f} - {ragas.context_precision_reason[:80]}")
            logger.info(f"  Ctx Recall:   {ragas.context_recall:.2f} - {ragas.context_recall_reason[:80]}")
            logger.info(f"  Correctness:  {ragas.answer_correctness:.2f} - {ragas.answer_correctness_reason[:80]}")
            logger.info(f"{'='*70}")
    
    def _query_graphmem(self, gm: GraphMem, query: str) -> Tuple[str, str, TokenMetrics, LatencyMetrics]:
        """Query GraphMem and return (answer, context, tokens, latency)."""
        tokens = TokenMetrics()
        latency = LatencyMetrics()
        
        total_start = time.perf_counter()
        
        try:
            response = gm.query(query)
            answer = response.answer
            
            # Get the context from response
            context = ""
            if hasattr(response, 'context') and response.context:
                if isinstance(response.context, list):
                    context = "\n\n".join(str(c) for c in response.context)
                else:
                    context = str(response.context)
            
            # Extract tokens
            if hasattr(response, 'context_tokens'):
                tokens.context_tokens = response.context_tokens
            else:
                tokens.context_tokens = len(context) // 4
            
            tokens.prompt_tokens = tokens.context_tokens + len(query) // 4
            tokens.completion_tokens = len(answer) // 4
            tokens.total_tokens = tokens.prompt_tokens + tokens.completion_tokens
            
        except Exception as e:
            answer = f"Error: {e}"
            context = ""
            tokens.total_tokens = 100
        
        total_end = time.perf_counter()
        latency.total_ms = (total_end - total_start) * 1000
        latency.retrieval_ms = latency.total_ms * 0.3
        latency.llm_generation_ms = latency.total_ms * 0.7
        
        return answer, context, tokens, latency
    
    def _calculate_system_metrics(self, results: SystemResults):
        """Calculate summary metrics including RAGAS averages."""
        if not results.queries:
            return
        
        results.total_queries = len(results.queries)
        results.correct_queries = sum(1 for r in results.queries if r.correct)
        
        if results.total_queries > 0:
            results.accuracy = results.correct_queries / results.total_queries
            
            # RAGAS averages
            results.avg_faithfulness = sum(r.ragas.faithfulness for r in results.queries) / results.total_queries
            results.avg_answer_relevancy = sum(r.ragas.answer_relevancy for r in results.queries) / results.total_queries
            results.avg_context_precision = sum(r.ragas.context_precision for r in results.queries) / results.total_queries
            results.avg_context_recall = sum(r.ragas.context_recall for r in results.queries) / results.total_queries
            results.avg_answer_correctness = sum(r.ragas.answer_correctness for r in results.queries) / results.total_queries
            results.avg_ragas_score = sum(r.ragas.overall_score for r in results.queries) / results.total_queries
        
        # Token metrics
        total_context = sum(r.tokens.context_tokens for r in results.queries)
        results.avg_context_tokens = total_context / results.total_queries if results.total_queries > 0 else 0
        results.total_tokens = sum(r.tokens.total_tokens for r in results.queries)
        
        # Cost metrics
        total_cost = sum(r.cost.total_cost for r in results.queries)
        results.total_cost_usd = total_cost
        results.cost_per_query = total_cost / results.total_queries if results.total_queries > 0 else 0
        
        # Latency metrics
        latencies = [r.latency.total_ms for r in results.queries]
        if latencies:
            results.avg_latency_ms = sum(latencies) / len(latencies)
            sorted_lat = sorted(latencies)
            results.p50_latency_ms = sorted_lat[len(sorted_lat) // 2]
            results.p95_latency_ms = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
    
    def run(
        self,
        n_corpus_docs: int = 50,
        n_qa_samples: int = 30,
        seed: int = 42,
    ) -> Tuple[SystemResults, SystemResults]:
        """Run comprehensive evaluation with RAGAS metrics."""
        random.seed(seed)
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE EVALUATION WITH RAGAS METRICS")
        print("=" * 80)
        print(f"   Corpus: {len(self.corpus)} docs (using {n_corpus_docs})")
        print(f"   QA Samples: {len(self.qa_samples)} (testing {n_qa_samples})")
        print(f"   Provider: {self.provider}")
        print(f"   LLM: {self.llm_model}")
        print(f"   Embeddings: {self.embedding_model}")
        print(f"   Backend: Turso ({self.turso_db_path})")
        print(f"   RAGAS Eval: {self.use_ragas}")
        print(f"   Debug mode: {self.debug}")
        print("=" * 80)
        
        # Validate data
        if not self.corpus:
            logger.error("❌ No corpus documents!")
            return self.graphmem_results, self.naive_results
        
        if not self.qa_samples:
            logger.error("❌ No QA samples!")
            return self.graphmem_results, self.naive_results
        
        # Sample data
        corpus_sample = random.sample(self.corpus, min(n_corpus_docs, len(self.corpus)))
        qa_sample = random.sample(self.qa_samples, min(n_qa_samples, len(self.qa_samples)))
        
        # ====================== PHASE 1: INGESTION ======================
        print("\n📦 PHASE 1: Ingesting corpus...")
        
        gm = self._init_graphmem()
        naive = self._init_naive_rag()
        
        # Initialize RAGAS evaluator
        if self.use_ragas:
            self._init_ragas_evaluator()
        
        if self.skip_ingestion:
            logger.info("⏭️  Skipping ingestion (--skip-ingestion)")
        else:
            if os.path.exists(self.turso_db_path):
                os.remove(self.turso_db_path)
                logger.info(f"   Removed existing {self.turso_db_path}")
            
            gm = self._init_graphmem()
            
            ingestion_start = time.perf_counter()
            
            documents = []
            for i, doc in enumerate(corpus_sample):
                content = f"{doc.get('title', '')}\n{doc.get('body', doc.get('content', ''))}"
                doc_id = f"doc_{i}"
                documents.append({"id": doc_id, "content": content[:8000]})
            
            print(f"   📊 GraphMem: Ingesting {len(documents)} documents...")
            gm_result = gm.ingest_batch(
                documents=documents,
                max_workers=10,
                show_progress=True,
                aggressive=True,
            )
            logger.info(f"   GraphMem: {gm_result.get('documents_processed', 0)} docs, "
                       f"{gm_result.get('total_entities', 0)} entities")
            
            print(f"   📊 NaiveRAG: Ingesting {len(documents)} documents...")
            for doc in documents:
                naive.ingest(doc["content"], doc["id"])
            logger.info(f"   NaiveRAG: {len(naive.chunks)} chunks")
            
            ingestion_time = time.perf_counter() - ingestion_start
            logger.info(f"   Total ingestion time: {ingestion_time:.1f}s")
        
        # Get memory metrics
        try:
            memory = gm._memory
            self.graphmem_results.memory.total_documents = n_corpus_docs
            self.graphmem_results.memory.total_entities = len(memory.nodes)
            self.graphmem_results.memory.total_relationships = len(memory.edges)
            self.graphmem_results.memory.total_clusters = len(memory.clusters)
            
            entity_names = set(n.name.lower() for n in memory.nodes.values())
            self.graphmem_results.memory.unique_entity_names = len(entity_names)
            
            if self.graphmem_results.memory.total_entities > 0:
                self.graphmem_results.memory.deduplication_ratio = (
                    self.graphmem_results.memory.unique_entity_names / 
                    self.graphmem_results.memory.total_entities
                )
        except Exception as e:
            logger.warning(f"Could not get memory metrics: {e}")
        
        # ====================== PHASE 2: QUERIES WITH RAGAS ======================
        print(f"\n📊 PHASE 2: Running {len(qa_sample)} queries with RAGAS evaluation...")
        
        for i, qa in enumerate(qa_sample):
            query = qa['query']
            expected = qa['answer']
            q_type = self._get_question_type(qa)
            
            # ---- Query GraphMem ----
            gm_answer, gm_context, gm_tokens, gm_latency = self._query_graphmem(gm, query)
            gm_correct = self._simple_check_answer(gm_answer, expected)
            gm_cost = self._calculate_cost(gm_tokens, self.llm_model)
            
            # RAGAS evaluation
            if self.use_ragas and self.ragas_evaluator:
                gm_ragas = self.ragas_evaluator.evaluate_all(query, gm_context, gm_answer, expected)
            else:
                gm_ragas = RAGASMetrics()
            
            gm_result = QueryResult(
                query=query,
                expected=expected,
                predicted=gm_answer,
                context=gm_context,
                correct=gm_correct,
                question_type=q_type,
                ragas=gm_ragas,
                tokens=gm_tokens,
                latency=gm_latency,
                cost=gm_cost,
            )
            self.graphmem_results.queries.append(gm_result)
            self._update_accuracy_breakdown(self.graphmem_results, q_type, gm_correct)
            self._log_debug("GraphMem", query, expected, gm_answer, gm_context, gm_ragas, q_type)
            
            # ---- Query Naive RAG ----
            naive_answer, naive_context, naive_tokens, naive_latency = naive.query(query)
            naive_correct = self._simple_check_answer(naive_answer, expected)
            naive_cost = self._calculate_cost(naive_tokens, self.llm_model)
            
            # RAGAS evaluation
            if self.use_ragas and self.ragas_evaluator:
                naive_ragas = self.ragas_evaluator.evaluate_all(query, naive_context, naive_answer, expected)
            else:
                naive_ragas = RAGASMetrics()
            
            naive_result = QueryResult(
                query=query,
                expected=expected,
                predicted=naive_answer,
                context=naive_context,
                correct=naive_correct,
                question_type=q_type,
                ragas=naive_ragas,
                tokens=naive_tokens,
                latency=naive_latency,
                cost=naive_cost,
            )
            self.naive_results.queries.append(naive_result)
            self._update_accuracy_breakdown(self.naive_results, q_type, naive_correct)
            self._log_debug("NaiveRAG", query, expected, naive_answer, naive_context, naive_ragas, q_type)
            
            # Progress
            if (i + 1) % 5 == 0:
                gm_ragas_avg = sum(r.ragas.overall_score for r in self.graphmem_results.queries) / len(self.graphmem_results.queries)
                naive_ragas_avg = sum(r.ragas.overall_score for r in self.naive_results.queries) / len(self.naive_results.queries)
                logger.info(f"   [{i+1}/{len(qa_sample)}] GraphMem RAGAS: {gm_ragas_avg:.2f} | NaiveRAG RAGAS: {naive_ragas_avg:.2f}")
        
        # ====================== PHASE 3: CALCULATE METRICS ======================
        print("\n📈 PHASE 3: Calculating metrics...")
        
        self._calculate_system_metrics(self.graphmem_results)
        self._calculate_system_metrics(self.naive_results)
        
        # ====================== PHASE 4: PRINT RESULTS ======================
        self._print_comparison()
        self._save_results()
        
        return self.graphmem_results, self.naive_results
    
    def run_aligned(self, qa_index: int = 0):
        """
        Run aligned evaluation: pick 1 QA sample and ingest ONLY its evidence corpus.
        
        This ensures the QA question has matching evidence in the corpus.
        """
        print("\n" + "=" * 80)
        print("🎯 ALIGNED EVALUATION (1 QA + Matching Corpus)")
        print("=" * 80)
        
        # Validate data
        if not self.corpus:
            logger.error("❌ No corpus!")
            return
        if not self.qa_samples:
            logger.error("❌ No QA samples!")
            return
        
        if qa_index >= len(self.qa_samples):
            qa_index = 0
        
        # Pick the QA sample
        qa_sample = self.qa_samples[qa_index]
        query = qa_sample['query']
        expected = qa_sample['answer']
        q_type = self._get_question_type(qa_sample)
        
        print(f"\n📝 Selected QA (index {qa_index}):")
        print(f"   Question: {query[:100]}...")
        print(f"   Expected: {expected}")
        print(f"   Type: {q_type}")
        
        # Find aligned corpus docs
        aligned_corpus = self._find_aligned_corpus(qa_sample)
        
        if not aligned_corpus:
            # Try with more relaxed matching - just use the evidence facts directly
            print("   ⚠️ No exact corpus matches found, using evidence directly")
            aligned_corpus = []
            for ev in qa_sample.get("evidence_list", []):
                aligned_corpus.append({
                    "title": ev.get("title", ""),
                    "body": ev.get("fact", ""),
                    "source": ev.get("source", ""),
                })
        
        print(f"   📚 Aligned corpus: {len(aligned_corpus)} documents")
        for doc in aligned_corpus:
            print(f"      - {doc.get('title', 'untitled')[:60]}...")
        
        print("=" * 80)
        
        # ====================== PHASE 1: INGEST ALIGNED CORPUS ======================
        gm = self._init_graphmem()
        naive = self._init_naive_rag()
        
        if self.skip_ingestion:
            print("\n📦 PHASE 1: Skipping ingestion (--skip-ingestion)")
            logger.info("   Using existing data in Turso DB")
            # Get memory metrics from existing data
            try:
                memory = gm._memory
                self.graphmem_results.memory.total_documents = len(aligned_corpus)
                self.graphmem_results.memory.total_entities = len(memory.nodes)
                self.graphmem_results.memory.total_relationships = len(memory.edges)
                self.graphmem_results.memory.total_clusters = len(memory.clusters)
                logger.info(f"   GraphMem: {len(memory.nodes)} entities, {len(memory.edges)} relationships")
            except Exception as e:
                logger.warning(f"Could not get memory metrics: {e}")
        else:
            print("\n📦 PHASE 1: Ingesting aligned corpus...")
            
            # Clean up existing Turso DB for fresh ingestion
            if os.path.exists(self.turso_db_path):
                os.remove(self.turso_db_path)
                gm = self._init_graphmem()  # Re-init after DB delete
            
            # Prepare documents
            documents = []
            for i, doc in enumerate(aligned_corpus):
                content = f"{doc.get('title', '')}\n{doc.get('body', doc.get('content', ''))}"
                doc_id = f"aligned_doc_{i}"
                documents.append({"id": doc_id, "content": content[:8000]})
            
            # GraphMem ingest
            print(f"   📊 GraphMem: Ingesting {len(documents)} documents...")
            gm_result = gm.ingest_batch(
                documents=documents,
                max_workers=10,
                show_progress=True,
                aggressive=True,
            )
            logger.info(f"   GraphMem: {gm_result.get('documents_processed', 0)} docs, "
                       f"{gm_result.get('total_entities', 0)} entities")
            
            # Naive RAG ingest
            print(f"   📊 NaiveRAG: Ingesting {len(documents)} documents...")
            for doc in documents:
                naive.ingest(doc["content"], doc["id"])
            logger.info(f"   NaiveRAG: {len(naive.chunks)} chunks")
            
            # Get memory metrics
            try:
                memory = gm._memory
                self.graphmem_results.memory.total_documents = len(documents)
                self.graphmem_results.memory.total_entities = len(memory.nodes)
                self.graphmem_results.memory.total_relationships = len(memory.edges)
                self.graphmem_results.memory.total_clusters = len(memory.clusters)
            except Exception as e:
                logger.warning(f"Could not get memory metrics: {e}")
        
        # Initialize RAGAS evaluator
        if self.use_ragas:
            self._init_ragas_evaluator()
        
        # ====================== PHASE 2: QUERY ======================
        print(f"\n📊 PHASE 2: Running query...")
        
        # ---- Query GraphMem ----
        print("\n   🧠 GraphMem:")
        gm_answer, gm_context, gm_tokens, gm_latency = self._query_graphmem(gm, query)
        gm_correct = self._simple_check_answer(gm_answer, expected)
        gm_cost = self._calculate_cost(gm_tokens, self.llm_model)
        
        # RAGAS evaluation
        if self.use_ragas and self.ragas_evaluator:
            print("   📏 Running RAGAS evaluation...")
            gm_ragas = self.ragas_evaluator.evaluate_all(query, gm_context, gm_answer, expected)
        else:
            gm_ragas = RAGASMetrics()
        
        gm_result = QueryResult(
            query=query,
            expected=expected,
            predicted=gm_answer,
            context=gm_context,
            correct=gm_correct,
            question_type=q_type,
            ragas=gm_ragas,
            tokens=gm_tokens,
            latency=gm_latency,
            cost=gm_cost,
        )
        self.graphmem_results.queries.append(gm_result)
        self._log_debug("GraphMem", query, expected, gm_answer, gm_context, gm_ragas, q_type)
        
        # ---- Query Naive RAG ----
        print("\n   📚 NaiveRAG:")
        naive_answer, naive_context, naive_tokens, naive_latency = naive.query(query)
        naive_correct = self._simple_check_answer(naive_answer, expected)
        naive_cost = self._calculate_cost(naive_tokens, self.llm_model)
        
        # RAGAS evaluation
        if self.use_ragas and self.ragas_evaluator:
            naive_ragas = self.ragas_evaluator.evaluate_all(query, naive_context, naive_answer, expected)
        else:
            naive_ragas = RAGASMetrics()
        
        naive_result = QueryResult(
            query=query,
            expected=expected,
            predicted=naive_answer,
            context=naive_context,
            correct=naive_correct,
            question_type=q_type,
            ragas=naive_ragas,
            tokens=naive_tokens,
            latency=naive_latency,
            cost=naive_cost,
        )
        self.naive_results.queries.append(naive_result)
        self._log_debug("NaiveRAG", query, expected, naive_answer, naive_context, naive_ragas, q_type)
        
        # ====================== PHASE 3: CALCULATE & PRINT ======================
        print("\n📈 PHASE 3: Results...")
        
        self._calculate_system_metrics(self.graphmem_results)
        self._calculate_system_metrics(self.naive_results)
        
        self._print_comparison()
        self._save_results()
        
        return self.graphmem_results, self.naive_results
    
    def _print_comparison(self):
        """Print side-by-side comparison with RAGAS metrics."""
        gm = self.graphmem_results
        naive = self.naive_results
        
        print("\n" + "=" * 80)
        print("📊 RESULTS: GraphMem vs Naive RAG (RAGAS Evaluation)")
        print("=" * 80)
        
        # RAGAS SCORES
        print("\n🎯 RAGAS SCORES (0-1, higher is better)")
        print("-" * 70)
        print(f"{'Metric':<25} {'GraphMem':>20} {'NaiveRAG':>20}")
        print("-" * 70)
        print(f"{'Faithfulness':<25} {gm.avg_faithfulness:>20.3f} {naive.avg_faithfulness:>20.3f}")
        print(f"{'Answer Relevancy':<25} {gm.avg_answer_relevancy:>20.3f} {naive.avg_answer_relevancy:>20.3f}")
        print(f"{'Context Precision':<25} {gm.avg_context_precision:>20.3f} {naive.avg_context_precision:>20.3f}")
        print(f"{'Context Recall':<25} {gm.avg_context_recall:>20.3f} {naive.avg_context_recall:>20.3f}")
        print(f"{'Answer Correctness':<25} {gm.avg_answer_correctness:>20.3f} {naive.avg_answer_correctness:>20.3f}")
        print("-" * 70)
        print(f"{'OVERALL RAGAS SCORE':<25} {gm.avg_ragas_score:>20.3f} {naive.avg_ragas_score:>20.3f}")
        
        # LEGACY ACCURACY
        print("\n📋 LEGACY ACCURACY (string matching)")
        print("-" * 70)
        print(f"{'Accuracy':<25} {gm.accuracy:>19.1%} {naive.accuracy:>19.1%}")
        print(f"{'Correct / Total':<25} {f'{gm.correct_queries}/{gm.total_queries}':>20} {f'{naive.correct_queries}/{naive.total_queries}':>20}")
        
        # EFFICIENCY
        print("\n💰 TOKEN & COST EFFICIENCY")
        print("-" * 70)
        print(f"{'Avg Context Tokens':<25} {gm.avg_context_tokens:>20.0f} {naive.avg_context_tokens:>20.0f}")
        print(f"{'Total Tokens':<25} {gm.total_tokens:>20,} {naive.total_tokens:>20,}")
        print(f"{'Total Cost (USD)':<25} ${gm.total_cost_usd:>19.4f} ${naive.total_cost_usd:>19.4f}")
        
        # LATENCY
        print("\n⏱️  LATENCY (ms)")
        print("-" * 70)
        print(f"{'Average':<25} {gm.avg_latency_ms:>20.0f} {naive.avg_latency_ms:>20.0f}")
        print(f"{'P50':<25} {gm.p50_latency_ms:>20.0f} {naive.p50_latency_ms:>20.0f}")
        print(f"{'P95':<25} {gm.p95_latency_ms:>20.0f} {naive.p95_latency_ms:>20.0f}")
        
        # MEMORY
        print("\n🧠 GRAPHMEM MEMORY METRICS")
        print("-" * 70)
        m = gm.memory
        print(f"{'Documents':<25} {m.total_documents:>20,}")
        print(f"{'Entities':<25} {m.total_entities:>20,}")
        print(f"{'Relationships':<25} {m.total_relationships:>20,}")
        print(f"{'Clusters':<25} {m.total_clusters:>20,}")
        print(f"{'Deduplication Ratio':<25} {m.deduplication_ratio:>20.3f}")
        
        # WINNER
        print("\n" + "=" * 80)
        print("🏆 WINNER")
        print("=" * 80)
        
        if gm.avg_ragas_score > naive.avg_ragas_score:
            improvement = ((gm.avg_ragas_score - naive.avg_ragas_score) / naive.avg_ragas_score * 100) if naive.avg_ragas_score > 0 else 0
            print(f"✅ GraphMem wins with {improvement:.1f}% higher RAGAS score!")
        elif naive.avg_ragas_score > gm.avg_ragas_score:
            improvement = ((naive.avg_ragas_score - gm.avg_ragas_score) / gm.avg_ragas_score * 100) if gm.avg_ragas_score > 0 else 0
            print(f"❌ Naive RAG wins with {improvement:.1f}% higher RAGAS score")
        else:
            print("🤝 It's a tie!")
        
        print("=" * 80)
    
    def _save_results(self):
        """Save results to JSON files."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "provider": self.provider,
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "use_ragas": self.use_ragas,
            },
            "graphmem": {
                "ragas_overall": self.graphmem_results.avg_ragas_score,
                "faithfulness": self.graphmem_results.avg_faithfulness,
                "answer_relevancy": self.graphmem_results.avg_answer_relevancy,
                "context_precision": self.graphmem_results.avg_context_precision,
                "context_recall": self.graphmem_results.avg_context_recall,
                "answer_correctness": self.graphmem_results.avg_answer_correctness,
                "legacy_accuracy": self.graphmem_results.accuracy,
                "total_queries": self.graphmem_results.total_queries,
                "avg_context_tokens": self.graphmem_results.avg_context_tokens,
                "total_cost_usd": self.graphmem_results.total_cost_usd,
                "avg_latency_ms": self.graphmem_results.avg_latency_ms,
            },
            "naive_rag": {
                "ragas_overall": self.naive_results.avg_ragas_score,
                "faithfulness": self.naive_results.avg_faithfulness,
                "answer_relevancy": self.naive_results.avg_answer_relevancy,
                "context_precision": self.naive_results.avg_context_precision,
                "context_recall": self.naive_results.avg_context_recall,
                "answer_correctness": self.naive_results.avg_answer_correctness,
                "legacy_accuracy": self.naive_results.accuracy,
                "total_queries": self.naive_results.total_queries,
                "avg_context_tokens": self.naive_results.avg_context_tokens,
                "total_cost_usd": self.naive_results.total_cost_usd,
                "avg_latency_ms": self.naive_results.avg_latency_ms,
            },
        }
        
        summary_file = output_dir / f"ragas_comparison_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📄 Summary saved: {summary_file}")
        
        # Debug log
        debug_file = output_dir / f"debug_log_{timestamp}.json"
        with open(debug_file, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        logger.info(f"📄 Debug log saved: {debug_file}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive GraphMem vs Naive RAG with RAGAS Metrics")
    
    # Provider settings
    parser.add_argument("--provider", default="azure", help="LLM provider")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4.1-mini", help="Azure LLM deployment")
    parser.add_argument("--azure-embedding-deployment", default="text-embedding-ada-002", help="Azure embedding deployment")
    
    # Evaluation settings
    parser.add_argument("--corpus-docs", type=int, default=100, help="Number of corpus docs")
    parser.add_argument("--qa-samples", type=int, default=50, help="Number of QA samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip ingestion")
    parser.add_argument("--no-ragas", action="store_true", help="Disable RAGAS evaluation")
    parser.add_argument("--turso-db", default="eval_graphmem.db", help="Turso database path")
    parser.add_argument("--aligned", action="store_true", help="Use aligned mode: pick 1 QA and ingest only its evidence corpus")
    parser.add_argument("--qa-index", type=int, default=0, help="QA sample index to use in aligned mode")
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.azure_endpoint,
        llm_model=args.azure_deployment,
        embedding_model=args.azure_embedding_deployment,
        azure_deployment=args.azure_deployment,
        azure_embedding_deployment=args.azure_embedding_deployment,
        turso_db_path=args.turso_db,
        debug=args.debug,
        skip_ingestion=args.skip_ingestion,
        use_ragas=not args.no_ragas,
    )
    
    # Aligned mode: use 1 QA and its matching corpus
    if args.aligned:
        evaluator.run_aligned(qa_index=args.qa_index)
    else:
        evaluator.run(
            n_corpus_docs=args.corpus_docs,
            n_qa_samples=args.qa_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

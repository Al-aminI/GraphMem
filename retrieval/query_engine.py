"""
GraphMem Query Engine

Orchestrates memory querying:
1. Retrieves relevant context
2. Generates answers using LLM
3. Aggregates from multiple communities
"""

from __future__ import annotations
import logging
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryQuery,
    MemoryResponse,
)

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine for memory retrieval and answer generation.
    
    Features:
    - Multi-hop reasoning over knowledge graph
    - Community-based answer aggregation
    - Confidence scoring
    - Parallel query processing
    """
    
    def __init__(
        self,
        llm,
        retriever,
        community_detector,
        context_engine,
        max_workers: int = 4,
    ):
        """
        Initialize query engine.
        
        Args:
            llm: LLM provider
            retriever: Memory retriever
            community_detector: Community detector for summaries
            context_engine: Context engineering engine
            max_workers: Parallel workers for community queries
        """
        self.llm = llm
        self.retriever = retriever
        self.community_detector = community_detector
        self.context_engine = context_engine
        self.max_workers = max_workers
    
    def query(
        self,
        query: MemoryQuery,
        memory: Memory,
    ) -> MemoryResponse:
        """
        Execute a query against memory.
        
        Args:
            query: Query specification
            memory: Memory to query
        
        Returns:
            MemoryResponse with answer and supporting context
        """
        start_time = time.time()
        
        # Retrieve relevant context
        retrieval_result = self.retriever.retrieve(query, memory)
        
        nodes = retrieval_result["nodes"]
        edges = retrieval_result["edges"]
        clusters = retrieval_result["clusters"]
        context = retrieval_result["context"]
        
        if not nodes and not clusters:
            return MemoryResponse(
                query=query.query,
                answer="No relevant information found in memory.",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        # Generate answers from communities
        community_answers = self._query_communities(
            query=query.query,
            clusters=clusters,
            nodes=nodes,
            edges=edges,
        )
        
        # Aggregate answers
        if community_answers:
            answer, confidence = self._aggregate_answers(
                query=query.query,
                answers=community_answers,
            )
        else:
            # Direct answer from context
            answer, confidence = self._generate_direct_answer(
                query=query.query,
                context=context,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MemoryResponse(
            query=query.query,
            answer=answer,
            confidence=confidence,
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            context=context,
            latency_ms=latency_ms,
        )
    
    def _query_communities(
        self,
        query: str,
        clusters: List[MemoryCluster],
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
    ) -> List[Dict[str, Any]]:
        """Query each relevant community for answers."""
        if not clusters:
            return []
        
        answers = []
        
        # Query communities in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._query_single_community,
                    query,
                    cluster,
                    nodes,
                    edges,
                ): cluster.id
                for cluster in clusters
            }
            
            for future in as_completed(futures):
                cluster_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        answers.append(result)
                except Exception as e:
                    logger.error(f"Community {cluster_id} query failed: {e}")
        
        return answers
    
    def _query_single_community(
        self,
        query: str,
        cluster: MemoryCluster,
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
    ) -> Optional[Dict[str, Any]]:
        """Query a single community for an answer."""
        # Get entities in this cluster
        cluster_nodes = [n for n in nodes if n.name in cluster.entities]
        cluster_edges = [
            e for e in edges
            if any(n.id in (e.source_id, e.target_id) for n in cluster_nodes)
        ]
        
        # Build context
        entity_context = self._format_entities(cluster_nodes)
        rel_context = self._format_relationships(cluster_edges)
        
        prompt = f"""You are answering questions using knowledge from a memory system.

Community Summary:
{cluster.summary}

Entity Details:
{entity_context}

Relationships:
{rel_context}

Question: {query}

Provide a direct answer based on the information above. If the information doesn't contain relevant details, say so.

Respond in JSON format:
{{"answer": "your answer", "confidence": 0-10}}"""
        
        try:
            response = self.llm.complete(prompt)
            parsed = self._parse_answer_response(response)
            if parsed:
                parsed["cluster_id"] = cluster.id
            return parsed
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return None
    
    def _aggregate_answers(
        self,
        query: str,
        answers: List[Dict[str, Any]],
    ) -> tuple:
        """Aggregate answers from multiple communities."""
        if not answers:
            return "No answer found.", 0.0
        
        # Sort by confidence
        sorted_answers = sorted(
            answers,
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )
        
        # If single high-confidence answer, return it
        if len(sorted_answers) == 1 or sorted_answers[0]["confidence"] >= 9:
            return sorted_answers[0]["answer"], sorted_answers[0]["confidence"] / 10.0
        
        # Aggregate multiple answers
        combined = "\n".join([
            f"Answer {i+1} (confidence {a['confidence']}/10): {a['answer']}"
            for i, a in enumerate(sorted_answers[:5])
        ])
        
        prompt = f"""Synthesize these answers into a single coherent response:

{combined}

Question: {query}

Provide the best synthesized answer:"""
        
        try:
            final_answer = self.llm.complete(prompt)
            avg_confidence = sum(a["confidence"] for a in sorted_answers[:3]) / min(3, len(sorted_answers))
            return final_answer.strip(), avg_confidence / 10.0
        except Exception as e:
            logger.error(f"Answer aggregation failed: {e}")
            return sorted_answers[0]["answer"], sorted_answers[0]["confidence"] / 10.0
    
    def _generate_direct_answer(
        self,
        query: str,
        context: str,
    ) -> tuple:
        """Generate answer directly from context."""
        prompt = f"""Answer the question using the provided context.

Context:
{context}

Question: {query}

Answer (be direct and concise):"""
        
        try:
            answer = self.llm.complete(prompt)
            return answer.strip(), 0.7  # Medium confidence for direct answers
        except Exception as e:
            logger.error(f"Direct answer generation failed: {e}")
            return "Unable to generate answer.", 0.0
    
    def _format_entities(self, nodes: List[MemoryNode]) -> str:
        """Format entities for prompt."""
        lines = []
        for node in nodes[:10]:
            desc = node.description or "No description"
            lines.append(f"- {node.name} ({node.entity_type}): {desc[:200]}")
        return "\n".join(lines) if lines else "No entity details available."
    
    def _format_relationships(self, edges: List[MemoryEdge]) -> str:
        """Format relationships for prompt."""
        lines = []
        for edge in edges[:10]:
            lines.append(f"- {edge.source_id} --[{edge.relation_type}]--> {edge.target_id}")
        return "\n".join(lines) if lines else "No relationships available."
    
    def _parse_answer_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM answer response."""
        try:
            parsed = json.loads(response)
            return {
                "answer": parsed.get("answer", ""),
                "confidence": int(parsed.get("confidence", 5)),
            }
        except json.JSONDecodeError:
            # Try to extract JSON
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(response[start:end + 1])
                    return {
                        "answer": parsed.get("answer", ""),
                        "confidence": int(parsed.get("confidence", 5)),
                    }
                except:
                    pass
            
            # Fallback: use response as answer
            return {
                "answer": response.strip(),
                "confidence": 5,
            }


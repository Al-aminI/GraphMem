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
        
        # ===== PRIORITIZE DIRECT ENTITY ANSWERS =====
        # If we found specific entities, use them DIRECTLY instead of community summaries
        # This is critical for avoiding noise pollution from other entities
        
        answer = None
        confidence = 0.0
        
        # First try: Direct answer from retrieved entities (BEST for specific queries)
        if nodes and context:
            # Build entity-focused context
            entity_context = self._build_entity_context(nodes, edges)
            if entity_context:
                answer, confidence = self._generate_direct_answer(
                    query=query.query,
                    context=entity_context,
                )
                if confidence >= 0.5:
                    logger.debug(f"Using direct entity answer (confidence={confidence})")
        
        # Second try: Community answers (for broader queries)
        if not answer or confidence < 0.5:
            all_nodes = list(memory.nodes.values()) if memory.nodes else nodes
            all_edges = list(memory.edges.values()) if memory.edges else edges
            
        community_answers = self._query_communities(
            query=query.query,
            clusters=clusters,
                nodes=all_nodes,
                edges=all_edges,
        )
        
        if community_answers:
                community_answer, community_confidence = self._aggregate_answers(
                query=query.query,
                answers=community_answers,
            )
                # Use community answer if better than direct
                if community_confidence > confidence:
                    answer = community_answer
                    confidence = community_confidence
        
        # Fallback: General context answer
        if not answer:
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
        cluster_node_ids = {n.id for n in cluster_nodes}
        
        # Get edges involving cluster nodes
        cluster_edges = [
            e for e in edges
            if e.source_id in cluster_node_ids or e.target_id in cluster_node_ids
        ]
        
        # IMPORTANT: Also include connected nodes that aren't in this cluster
        # This ensures cross-cluster relationships are visible
        connected_node_ids = set()
        for e in cluster_edges:
            connected_node_ids.add(e.source_id)
            connected_node_ids.add(e.target_id)
        
        # Add connected nodes that aren't already in cluster
        all_relevant_nodes = list(cluster_nodes)
        for n in nodes:
            if n.id in connected_node_ids and n.id not in cluster_node_ids:
                all_relevant_nodes.append(n)
        
        # Build context with ALL relevant nodes (cluster + connected)
        entity_context = self._format_entities(all_relevant_nodes)
        rel_context = self._format_relationships(cluster_edges)
        
        prompt = f"""You are answering questions using knowledge from a memory system.

COMMUNITY SUMMARY:
{cluster.summary}

ENTITIES IN THIS COMMUNITY:
{entity_context}

RELATIONSHIPS:
{rel_context}

QUESTION: {query}

INSTRUCTIONS:
1. Consider ALL entities and relationships shown above
2. If multiple facts are relevant to the question, include them ALL
3. Cross-reference the community summary with the specific entities and relationships
4. Provide a comprehensive answer that covers all relevant information

Respond in JSON format:
{{"answer": "your comprehensive answer", "confidence": 0-10}}"""
        
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
        
        prompt = f"""Synthesize these answers from different knowledge communities into a single comprehensive response.

ANSWERS FROM DIFFERENT COMMUNITIES:
{combined}

QUESTION: {query}

INSTRUCTIONS:
1. Combine information from ALL answers, not just the highest confidence one
2. If answers mention different entities or facts, include ALL of them
3. Remove duplicates but preserve all unique information
4. Present a unified, comprehensive answer

Synthesized Answer:"""
        
        try:
            final_answer = self.llm.complete(prompt)
            avg_confidence = sum(a["confidence"] for a in sorted_answers[:3]) / min(3, len(sorted_answers))
            return final_answer.strip(), avg_confidence / 10.0
        except Exception as e:
            logger.error(f"Answer aggregation failed: {e}")
            return sorted_answers[0]["answer"], sorted_answers[0]["confidence"] / 10.0
    
    def _build_entity_context(
        self,
        nodes: List[MemoryNode],
        edges: List[MemoryEdge],
    ) -> str:
        """
        Build context DIRECTLY from retrieved entities.
        
        This is more precise than community summaries because it focuses
        on the specific entities found by retrieval, avoiding noise.
        """
        if not nodes:
            return ""
        
        context_parts = []
        
        # Entity details
        context_parts.append("## ENTITIES (directly relevant to your query)")
        for node in nodes[:20]:  # More entities
            aliases = ""
            if hasattr(node, 'aliases') and node.aliases:
                other = [a for a in node.aliases if a != node.name]
                if other:
                    aliases = f" [Also known as: {', '.join(other[:5])}]"
            
            desc = node.description or "No description"
            context_parts.append(f"• {node.name} ({node.entity_type}){aliases}")
            context_parts.append(f"  Description: {desc[:300]}")
        
        # Relationships
        if edges:
            context_parts.append("\n## RELATIONSHIPS")
            node_ids = {n.id for n in nodes}
            node_names = {n.name.lower(): n.name for n in nodes}
            
            for edge in edges[:30]:  # More relationships
                # Only include relationships involving our nodes
                source_match = edge.source_id in node_ids or edge.source_id.lower() in node_names
                target_match = edge.target_id in node_ids or edge.target_id.lower() in node_names
                
                if source_match or target_match:
                    temporal = ""
                    if hasattr(edge, 'valid_from') and edge.valid_from:
                        from_str = edge.valid_from.strftime("%Y") if hasattr(edge.valid_from, 'strftime') else str(edge.valid_from)
                        until_str = "present"
                        if hasattr(edge, 'valid_until') and edge.valid_until:
                            until_str = edge.valid_until.strftime("%Y") if hasattr(edge.valid_until, 'strftime') else str(edge.valid_until)
                        temporal = f" [valid: {from_str} → {until_str}]"
                    
                    context_parts.append(f"• {edge.source_id} --[{edge.relation_type}]--> {edge.target_id}{temporal}")
                    if edge.description:
                        context_parts.append(f"  Detail: {edge.description[:200]}")
        
        return "\n".join(context_parts)
    
    def _generate_direct_answer(
        self,
        query: str,
        context: str,
    ) -> tuple:
        """Generate answer directly from context."""
        prompt = f"""You are answering questions based on a knowledge graph memory system.

CONTEXT (contains entities, relationships, temporal info, and topic summaries):
{context}

QUESTION: {query}

CRITICAL INSTRUCTIONS:
1. **ALIASES**: Entities may have multiple names (e.g., "Alexander Chen" = "Dr. Chen" = "The Quantum Pioneer"). 
   If the question uses any alias, find the matching entity by ANY of its names.
   
2. **TEMPORAL VALIDITY**: Relationships have time periods (valid_from → valid_until).
   - If asked "WHO IS current X" → find relationship with valid_until = present/None
   - If asked "WHO WAS X in YEAR" → find relationship where YEAR is between valid_from and valid_until
   - "Present" or no valid_until means the relationship is CURRENT
   
3. **EXHAUSTIVE SEARCH**: Check ALL entities and relationships, not just the first match.

4. **SYNTHESIZE**: Combine information from multiple sources if relevant.

5. **BE SPECIFIC**: If you find temporal info, include it (e.g., "X was CEO from 2015 to 2018").

Answer:"""
        
        try:
            answer = self.llm.complete(prompt)
            return answer.strip(), 0.7  # Medium confidence for direct answers
        except Exception as e:
            logger.error(f"Direct answer generation failed: {e}")
            return "Unable to generate answer.", 0.0
    
    def _format_entities(self, nodes: List[MemoryNode]) -> str:
        """Format entities for prompt, including aliases."""
        lines = []
        for node in nodes[:15]:  # Show more entities
            desc = node.description or "No description"
            # Include aliases if available
            aliases_str = ""
            if hasattr(node, 'aliases') and node.aliases:
                other_aliases = [a for a in node.aliases if a != node.name]
                if other_aliases:
                    aliases_str = f" [also known as: {', '.join(other_aliases[:5])}]"
            lines.append(f"- {node.name} ({node.entity_type}){aliases_str}: {desc[:200]}")
        return "\n".join(lines) if lines else "No entity details available."
    
    def _format_relationships(self, edges: List[MemoryEdge]) -> str:
        """Format relationships for prompt, including temporal validity."""
        lines = []
        for edge in edges[:15]:  # Show more relationships
            # Include temporal validity if available
            temporal_str = ""
            if hasattr(edge, 'valid_from') and edge.valid_from:
                from_str = edge.valid_from.strftime("%Y") if isinstance(edge.valid_from, datetime) else str(edge.valid_from)
                if hasattr(edge, 'valid_until') and edge.valid_until:
                    until_str = edge.valid_until.strftime("%Y") if isinstance(edge.valid_until, datetime) else str(edge.valid_until)
                else:
                    until_str = "present"
                temporal_str = f" [{from_str} → {until_str}]"
            
            lines.append(f"- {edge.source_id} --[{edge.relation_type}]--> {edge.target_id}{temporal_str}")
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


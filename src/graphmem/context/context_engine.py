"""
GraphMem Context Engine

Super context engineering for optimal LLM context construction.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from graphmem.context.chunker import DocumentChunk, DocumentChunker
from graphmem.context.multimodal import MultiModalProcessor, MultiModalInput, ProcessedDocument
from graphmem.core.memory_types import MemoryNode, MemoryEdge, MemoryCluster

logger = logging.getLogger(__name__)


def _get_importance_value(importance) -> float:
    """Safely get numeric value from importance (enum or float)."""
    if hasattr(importance, 'value'):
        return importance.value
    return float(importance) if importance is not None else 5.0


@dataclass
class ContextWindow:
    """Represents a constructed context window."""
    content: str
    tokens_used: int
    token_limit: int
    
    # Components included
    entities: List[MemoryNode] = field(default_factory=list)
    relationships: List[MemoryEdge] = field(default_factory=list)
    communities: List[MemoryCluster] = field(default_factory=list)
    documents: List[DocumentChunk] = field(default_factory=list)
    
    # Metadata
    query: Optional[str] = None
    priority_order: List[str] = field(default_factory=list)
    truncated: bool = False


class ContextEngine:
    """
    Super context engineering for GraphMem.
    
    Features:
    - Optimal context window construction
    - Priority-based content selection
    - Token budget management
    - Multi-modal context integration
    - Dynamic context expansion
    """
    
    def __init__(
        self,
        llm=None,
        embeddings=None,
        token_limit: int = 8000,
        tokens_per_char: float = 0.25,  # Approximate
    ):
        """
        Initialize context engine.
        
        Args:
            llm: LLM provider (for summarization)
            embeddings: Embedding provider
            token_limit: Maximum context tokens
            tokens_per_char: Tokens per character ratio
        """
        self.llm = llm
        self.embeddings = embeddings
        self.token_limit = token_limit
        self.tokens_per_char = tokens_per_char
        
        self.multimodal = MultiModalProcessor(llm=llm)
        self.chunker = DocumentChunker()
    
    def build_context(
        self,
        query: str,
        entities: List[MemoryNode] = None,
        relationships: List[MemoryEdge] = None,
        communities: List[MemoryCluster] = None,
        documents: List[DocumentChunk] = None,
        priority: str = "balanced",
        include_query: bool = True,
    ) -> ContextWindow:
        """
        Build optimal context window.
        
        Args:
            query: Query for context
            entities: Relevant entities
            relationships: Relevant relationships
            communities: Relevant communities
            documents: Relevant documents
            priority: Priority mode (entities, relationships, communities, balanced)
            include_query: Include query in context
        
        Returns:
            ContextWindow with constructed context
        """
        entities = entities or []
        relationships = relationships or []
        communities = communities or []
        documents = documents or []
        
        # Define priority order
        priority_order = self._get_priority_order(priority)
        
        # Calculate available budget
        budget = self.token_limit
        if include_query:
            budget -= self._estimate_tokens(f"Query: {query}\n\n")
        
        # Allocate budget by priority
        allocations = self._allocate_budget(
            budget=budget,
            priority_order=priority_order,
            counts={
                "communities": len(communities),
                "entities": len(entities),
                "relationships": len(relationships),
                "documents": len(documents),
            },
        )
        
        # Build context sections
        sections = []
        used_tokens = 0
        truncated = False
        
        included_entities = []
        included_relationships = []
        included_communities = []
        included_documents = []
        
        for component in priority_order:
            if component == "communities" and communities:
                text, tokens, items = self._build_communities_section(
                    communities, allocations["communities"]
                )
                if text:
                    sections.append(("Topic Summaries", text))
                    used_tokens += tokens
                    included_communities = items
                    if tokens < allocations["communities"]:
                        truncated = True
            
            elif component == "entities" and entities:
                text, tokens, items = self._build_entities_section(
                    entities, allocations["entities"]
                )
                if text:
                    sections.append(("Relevant Entities", text))
                    used_tokens += tokens
                    included_entities = items
                    if tokens < allocations["entities"]:
                        truncated = True
            
            elif component == "relationships" and relationships:
                text, tokens, items = self._build_relationships_section(
                    relationships, allocations["relationships"]
                )
                if text:
                    sections.append(("Relationships", text))
                    used_tokens += tokens
                    included_relationships = items
                    if tokens < allocations["relationships"]:
                        truncated = True
            
            elif component == "documents" and documents:
                text, tokens, items = self._build_documents_section(
                    documents, allocations["documents"]
                )
                if text:
                    sections.append(("Supporting Documents", text))
                    used_tokens += tokens
                    included_documents = items
                    if tokens < allocations["documents"]:
                        truncated = True
        
        # Construct final context
        context_parts = []
        if include_query:
            context_parts.append(f"Query: {query}")
        
        for section_name, section_content in sections:
            context_parts.append(f"\n## {section_name}\n{section_content}")
        
        content = "\n".join(context_parts)
        final_tokens = self._estimate_tokens(content)
        
        return ContextWindow(
            content=content,
            tokens_used=final_tokens,
            token_limit=self.token_limit,
            entities=included_entities,
            relationships=included_relationships,
            communities=included_communities,
            documents=included_documents,
            query=query,
            priority_order=priority_order,
            truncated=truncated,
        )
    
    def _get_priority_order(self, priority: str) -> List[str]:
        """Get component priority order."""
        orders = {
            "communities": ["communities", "entities", "relationships", "documents"],
            "entities": ["entities", "relationships", "communities", "documents"],
            "relationships": ["relationships", "entities", "communities", "documents"],
            "documents": ["documents", "entities", "relationships", "communities"],
            "balanced": ["communities", "entities", "relationships", "documents"],
        }
        return orders.get(priority, orders["balanced"])
    
    def _allocate_budget(
        self,
        budget: int,
        priority_order: List[str],
        counts: Dict[str, int],
    ) -> Dict[str, int]:
        """Allocate token budget to components."""
        allocations = {}
        remaining = budget
        
        # Priority weights
        weights = {
            priority_order[0]: 0.35,
            priority_order[1]: 0.30,
            priority_order[2]: 0.20,
            priority_order[3]: 0.15,
        }
        
        for component in priority_order:
            if counts.get(component, 0) == 0:
                allocations[component] = 0
            else:
                allocations[component] = int(budget * weights.get(component, 0.2))
        
        return allocations
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return int(len(text) * self.tokens_per_char)
    
    def _build_communities_section(
        self,
        communities: List[MemoryCluster],
        budget: int,
    ) -> Tuple[str, int, List[MemoryCluster]]:
        """Build communities section."""
        lines = []
        used_tokens = 0
        included = []
        
        # Sort by importance
        sorted_communities = sorted(
            communities,
            key=lambda c: _get_importance_value(c.importance),
            reverse=True,
        )
        
        for community in sorted_communities:
            line = f"• {community.summary}"
            line_tokens = self._estimate_tokens(line + "\n")
            
            if used_tokens + line_tokens > budget:
                break
            
            lines.append(line)
            used_tokens += line_tokens
            included.append(community)
        
        return "\n".join(lines), used_tokens, included
    
    def _build_entities_section(
        self,
        entities: List[MemoryNode],
        budget: int,
    ) -> Tuple[str, int, List[MemoryNode]]:
        """Build entities section."""
        lines = []
        used_tokens = 0
        included = []
        
        # Sort by importance
        sorted_entities = sorted(
            entities,
            key=lambda e: _get_importance_value(e.importance),
            reverse=True,
        )
        
        for entity in sorted_entities:
            desc = entity.description or entity.name
            line = f"• {entity.name} ({entity.entity_type}): {desc[:200]}"
            line_tokens = self._estimate_tokens(line + "\n")
            
            if used_tokens + line_tokens > budget:
                break
            
            lines.append(line)
            used_tokens += line_tokens
            included.append(entity)
        
        return "\n".join(lines), used_tokens, included
    
    def _build_relationships_section(
        self,
        relationships: List[MemoryEdge],
        budget: int,
    ) -> Tuple[str, int, List[MemoryEdge]]:
        """Build relationships section."""
        lines = []
        used_tokens = 0
        included = []
        
        # Sort by weight/confidence
        sorted_rels = sorted(
            relationships,
            key=lambda r: r.weight * r.confidence,
            reverse=True,
        )
        
        for rel in sorted_rels:
            line = f"• {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}"
            if rel.description:
                line += f": {rel.description[:100]}"
            line_tokens = self._estimate_tokens(line + "\n")
            
            if used_tokens + line_tokens > budget:
                break
            
            lines.append(line)
            used_tokens += line_tokens
            included.append(rel)
        
        return "\n".join(lines), used_tokens, included
    
    def _build_documents_section(
        self,
        documents: List[DocumentChunk],
        budget: int,
    ) -> Tuple[str, int, List[DocumentChunk]]:
        """Build documents section."""
        parts = []
        used_tokens = 0
        included = []
        
        for doc in documents:
            content = doc.content[:500]  # Limit per doc
            line_tokens = self._estimate_tokens(content + "\n\n")
            
            if used_tokens + line_tokens > budget:
                break
            
            parts.append(content)
            used_tokens += line_tokens
            included.append(doc)
        
        return "\n\n".join(parts), used_tokens, included
    
    def ingest_document(
        self,
        content: str,
        modality: str = "text",
        source_uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessedDocument:
        """
        Ingest a document for memory.
        
        Args:
            content: Document content
            modality: Content modality
            source_uri: Source location
            metadata: Additional metadata
        
        Returns:
            ProcessedDocument with chunks
        """
        input_data = MultiModalInput(
            content=content,
            modality=modality,
            source_uri=source_uri,
            metadata=metadata or {},
        )
        
        return self.multimodal.process(input_data)
    
    def summarize_context(
        self,
        context: ContextWindow,
        max_length: int = 500,
    ) -> str:
        """
        Summarize context window.
        
        Args:
            context: Context to summarize
            max_length: Maximum summary length
        
        Returns:
            Summary string
        """
        if not self.llm:
            # Extractive summary
            lines = context.content.split("\n")
            summary_lines = [l for l in lines if l.strip() and not l.startswith("#")][:5]
            return " ".join(summary_lines)[:max_length]
        
        prompt = f"""Summarize the following context in {max_length} characters or less:

{context.content}

Summary:"""
        
        try:
            return self.llm.complete(prompt)[:max_length]
        except Exception as e:
            logger.error(f"Context summarization failed: {e}")
            return context.content[:max_length]

    def extract_from_url(self, url: str) -> str:
        """
        Extract content from a URL (webpage).
        
        Args:
            url: URL to extract content from
        
        Returns:
            Extracted text content
        
        Example:
            >>> text = engine.extract_from_url("https://example.com/article")
        """
        from graphmem.context.extractors import extract_webpage
        return extract_webpage(url)


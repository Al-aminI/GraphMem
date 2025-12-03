"""
GraphMem Knowledge Graph Builder

Extracts entities and relationships from text to build knowledge graphs.
Uses state-of-the-art LLM-based extraction with parallel processing.
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from graphmem.core.memory_types import MemoryNode, MemoryEdge
from graphmem.core.exceptions import ExtractionError

logger = logging.getLogger(__name__)


# State-of-the-art extraction prompt optimized for comprehensive knowledge extraction
EXTRACTION_PROMPT = """
-Goal-
Given a text document, exhaustively identify every entity and relationship.
Extract as many unique knowledge triplets as the text supports.
Treat {max_triplets} as a minimum target - continue extracting beyond it when more are available.

-Steps-
1. Identify EVERY entity mentioned. For each entity, capture:
   - entity_name: Canonical, capitalized name
   - entity_type: Specific type (person, organization, location, event, concept, product, etc.)
   - entity_description: Comprehensive description with ALL facts, attributes, roles, timelines, and quantitative values

Format: ("entity"$$$$<name>$$$$<type>$$$$<description>)

2. Identify EVERY relationship (explicit or implicit). For each:
   - source_entity: Name of source entity
   - target_entity: Name of target entity  
   - relation: Short, precise verb phrase
   - relationship_description: Exhaustive explanation with causes, effects, timelines, quantities

Format: ("relationship"$$$$<source>$$$$<target>$$$$<relation>$$$$<description>)

3. Continue until no further distinct entities or relationships remain.

-Text-
{text}

-Output-
"""


@dataclass
class ExtractionConfig:
    """Configuration for knowledge extraction."""
    chunk_size: int = 2048
    chunk_overlap: int = 200
    max_triplets_per_chunk: int = 40
    max_workers: int = 8
    retry_on_failure: bool = True
    max_retries: int = 3


class KnowledgeGraph:
    """
    Builds knowledge graphs from text using LLM-based extraction.
    
    Features:
    - Parallel chunk processing for speed
    - Entity resolution to merge duplicates
    - Rich relationship extraction
    - Progress callbacks for monitoring
    """
    
    def __init__(
        self,
        llm,
        embeddings,
        store=None,  # Optional - can work in-memory
        entity_resolver=None,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,
        max_triplets_per_chunk: int = 40,
        max_workers: int = 8,
    ):
        """
        Initialize knowledge graph builder.
        
        Args:
            llm: LLM provider for extraction
            embeddings: Embedding provider for entity matching
            store: Graph store for persistence
            entity_resolver: Entity resolver for deduplication
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            max_triplets_per_chunk: Target triplets per chunk
            max_workers: Parallel workers for extraction
        """
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.entity_resolver = entity_resolver
        self.config = ExtractionConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_triplets_per_chunk=max_triplets_per_chunk,
            max_workers=max_workers,
        )
    
    def extract(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """
        Extract entities and relationships from content.
        
        Args:
            content: Text content to process
            metadata: Metadata to attach to extracted elements
            memory_id: ID of parent memory
            progress_callback: Optional progress callback
        
        Returns:
            Tuple of (nodes, edges)
        """
        if not content or not content.strip():
            return [], []
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        logger.info(f"Split content into {len(chunks)} chunks")
        
        if progress_callback:
            progress_callback("chunking", 0.1)
        
        # Extract from each chunk in parallel
        all_entities = []
        all_relationships = []
        
        if len(chunks) == 1:
            # Single chunk - process directly
            entities, relationships = self._extract_from_chunk(
                chunks[0], metadata, memory_id
            )
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        else:
            # Multiple chunks - process in parallel
            completed = 0
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._extract_from_chunk, chunk, metadata, memory_id
                    ): i
                    for i, chunk in enumerate(chunks)
                }
                
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        entities, relationships = future.result()
                        all_entities.extend(entities)
                        all_relationships.extend(relationships)
                        completed += 1
                        
                        if progress_callback:
                            progress = 0.1 + (completed / len(chunks)) * 0.4
                            progress_callback("extracting", progress)
                            
                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx} extraction failed: {e}")
        
        if progress_callback:
            progress_callback("resolving", 0.5)
        
        # Resolve entity duplicates
        nodes = self.entity_resolver.resolve(all_entities, memory_id)
        
        # Update edges with canonical names
        edges = self._resolve_edge_entities(all_relationships, nodes)
        
        logger.info(f"Extracted {len(nodes)} unique entities, {len(edges)} relationships")
        
        return nodes, edges
    
    def _split_into_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks."""
        chunks = []
        
        # Simple sentence-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) < self.config.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks if chunks else [content]
    
    def _extract_from_chunk(
        self,
        chunk: str,
        metadata: Dict[str, Any],
        memory_id: str,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Extract entities and relationships from a single chunk."""
        prompt = EXTRACTION_PROMPT.format(
            max_triplets=self.config.max_triplets_per_chunk,
            text=chunk,
        )
        
        try:
            response = self.llm.complete(prompt)
            entities, relationships = self._parse_extraction_response(response)
            
            # Create nodes with embeddings for vector search
            nodes = []
            for name, entity_type, description in entities:
                # Generate embedding for the node (name + description)
                text_to_embed = f"{name}: {description}" if description else name
                try:
                    embedding = self.embeddings.embed_text(text_to_embed)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {name}: {e}")
                    embedding = None
                
                node = MemoryNode(
                    id="",  # Will be generated
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    embedding=embedding,  # Add embedding for vector search
                    properties={**metadata, "source_chunk": chunk[:200]},
                    memory_id=memory_id,
                )
                nodes.append(node)
            
            # Create edges
            edges = []
            for source, target, relation, description in relationships:
                edge = MemoryEdge(
                    id="",  # Will be generated
                    source_id=source,  # Will be resolved to canonical
                    target_id=target,
                    relation_type=relation,
                    description=description,
                    properties=metadata,
                    memory_id=memory_id,
                )
                edges.append(edge)
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk: {e}")
            if self.config.retry_on_failure:
                # Retry with simpler prompt
                return self._extract_fallback(chunk, metadata, memory_id)
            return [], []
    
    def _extract_fallback(
        self,
        chunk: str,
        metadata: Dict[str, Any],
        memory_id: str,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Fallback extraction with simpler prompt."""
        try:
            prompt = f"""Extract key entities and relationships from this text.
            
Text: {chunk}

List entities as: ENTITY: name | type | description
List relationships as: RELATIONSHIP: source -> relation -> target | description"""
            
            response = self.llm.complete(prompt)
            
            # Parse simpler format
            entities = []
            relationships = []
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('ENTITY:'):
                    parts = line[7:].split('|')
                    if len(parts) >= 3:
                        entities.append((
                            parts[0].strip(),
                            parts[1].strip(),
                            parts[2].strip(),
                        ))
                elif line.startswith('RELATIONSHIP:'):
                    # Parse: source -> relation -> target | description
                    content = line[13:]
                    if '|' in content:
                        rel_part, desc = content.split('|', 1)
                        arrow_parts = rel_part.split('->')
                        if len(arrow_parts) >= 3:
                            relationships.append((
                                arrow_parts[0].strip(),
                                arrow_parts[2].strip(),
                                arrow_parts[1].strip(),
                                desc.strip(),
                            ))
            
            nodes = []
            for name, etype, desc in entities:
                # Generate embedding for vector search
                text_to_embed = f"{name}: {desc}" if desc else name
                try:
                    embedding = self.embeddings.embed_text(text_to_embed)
                except:
                    embedding = None
                    
                nodes.append(MemoryNode(
                    id="", name=name, entity_type=etype,
                    description=desc, embedding=embedding,
                    properties=metadata, memory_id=memory_id,
                ))
            
            edges = [
                MemoryEdge(
                    id="", source_id=src, target_id=tgt,
                    relation_type=rel, description=desc,
                    properties=metadata, memory_id=memory_id,
                )
                for src, tgt, rel, desc in relationships
            ]
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Fallback extraction also failed: {e}")
            return [], []
    
    def _parse_extraction_response(
        self,
        response: str,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Parse LLM extraction response."""
        entities = []
        relationships = []
        
        # Try primary format first
        entity_pattern = r'\("entity"\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\)'
        rel_pattern = r'\("relationship"\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\$\$\$\$"?(.+?)"?\)'
        
        entities = re.findall(entity_pattern, response)
        relationships = re.findall(rel_pattern, response)
        
        # Try alternative format
        if not entities and not relationships:
            entity_pattern_alt = r'\(entity\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\)'
            rel_pattern_alt = r'\(relationship\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\$\$\$\$(.+?)\)'
            
            entities = re.findall(entity_pattern_alt, response)
            relationships = re.findall(rel_pattern_alt, response)
        
        # Try line-based parsing
        if not entities and not relationships:
            for line in response.split('\n'):
                line = line.strip()
                if 'entity' in line.lower() and '$$$$' in line:
                    parts = line.split('$$$$')
                    if len(parts) >= 4:
                        entities.append((
                            parts[1].strip().strip('"'),
                            parts[2].strip().strip('"'),
                            parts[3].strip().strip('"'),
                        ))
                elif 'relationship' in line.lower() and '$$$$' in line:
                    parts = line.split('$$$$')
                    if len(parts) >= 5:
                        relationships.append((
                            parts[1].strip().strip('"'),
                            parts[2].strip().strip('"'),
                            parts[3].strip().strip('"'),
                            parts[4].strip().strip('"'),
                        ))
        
        logger.debug(f"Parsed {len(entities)} entities, {len(relationships)} relationships")
        return entities, relationships
    
    def _resolve_edge_entities(
        self,
        edges: List[MemoryEdge],
        nodes: List[MemoryNode],
    ) -> List[MemoryEdge]:
        """Update edge source/target to use canonical entity IDs."""
        # Build name to ID mapping
        name_to_id = {}
        for node in nodes:
            name_to_id[node.name.lower()] = node.id
            if node.canonical_name:
                name_to_id[node.canonical_name.lower()] = node.id
            for alias in node.aliases:
                name_to_id[alias.lower()] = node.id
        
        resolved_edges = []
        for edge in edges:
            source_id = name_to_id.get(edge.source_id.lower())
            target_id = name_to_id.get(edge.target_id.lower())
            
            if source_id and target_id:
                edge.source_id = source_id
                edge.target_id = target_id
                resolved_edges.append(edge)
            else:
                logger.debug(f"Could not resolve edge: {edge.source_id} -> {edge.target_id}")
        
        return resolved_edges


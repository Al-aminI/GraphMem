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


# State-of-the-art EXHAUSTIVE extraction prompt
# Captures EVERY piece of information from the text
EXTRACTION_PROMPT = """
You are an EXHAUSTIVE knowledge graph extractor. Your job is to capture EVERY SINGLE piece of information.

## CRITICAL: EXTRACT EVERYTHING

You MUST extract:
1. **ALL ENTITIES** - Every person, organization, product, location, concept, event, date, number mentioned
2. **ALL ATTRIBUTES** - Every property, value, characteristic of each entity
3. **ALL RELATIONSHIPS** - Every connection between entities, explicit or implicit
4. **ALL FACTS** - Every statement, claim, or piece of data in the text
5. **ALL NUMBERS** - Percentages, amounts, counts, measurements, statistics
6. **ALL DATES/TIMES** - Years, dates, periods, durations, deadlines
7. **ALL LOCATIONS** - Countries, cities, addresses, regions
8. **ALL ALIASES** - Nicknames, abbreviations, titles, alternate names

## ENTITY EXTRACTION RULES

1. **ALIASES ARE MANDATORY**: 
   - "Dr. Alexander Chen" → aliases="Dr. Alexander Chen, Dr. Chen, Alex Chen, A. Chen"
   - "Apple Inc." → aliases="Apple, Apple Computer, AAPL"
   - Include EVERY name variation mentioned OR commonly known
   
2. **RICH DESCRIPTIONS**: Include ALL facts about the entity in description
   - BAD: "A company" 
   - GOOD: "Technology company founded in 1976, headquartered in Cupertino, CA, market cap $3T, makes iPhone, Mac, iPad"

3. **ENTITY TYPES**: Use specific types
   - Person, Organization, Company, Product, Location, Event, Date, Concept, Technology, Number, Percentage, Amount

## RELATIONSHIP EXTRACTION RULES

1. **TEMPORAL VALIDITY**: Extract dates when mentioned
   - "was CEO from 2015 to 2018" → valid_from=2015, valid_until=2018
   - "since 2023" → valid_from=2023, valid_until=present
   - "in Q3 2024" → valid_from=2024-07, valid_until=2024-09

2. **IMPLICIT RELATIONSHIPS**: Extract implied connections
   - "Tesla's Elon Musk" implies "Musk leads Tesla"
   - "the $5M investment" implies an investment relationship

3. **ATTRIBUTE RELATIONSHIPS**: Create relationships for entity attributes
   - "Tesla has 100,000 employees" → (Tesla, has_employees, 100000)
   - "Apple's revenue is $400B" → (Apple, has_revenue, $400B)

## FORMAT

Entity: ("entity"$$$$<CANONICAL_NAME>$$$$<TYPE>$$$$<DETAILED_DESCRIPTION>$$$$<ALL_ALIASES>)
Relationship: ("relationship"$$$$<SOURCE>$$$$<TARGET>$$$$<RELATION>$$$$<DESCRIPTION>$$$$<VALID_FROM>$$$$<VALID_UNTIL>)

## EXAMPLE - EXHAUSTIVE EXTRACTION

Text: "In Q3 2024, Nvidia (NVDA) reported $35.1B revenue, up 94% year-over-year. CEO Jensen Huang announced the new Blackwell B200 GPU would ship in early 2025. The company now has over 26,000 employees worldwide."

Output:
("entity"$$$$Nvidia$$$$Company$$$$Semiconductor company specializing in GPUs and AI chips, reported $35.1B revenue in Q3 2024 with 94% YoY growth, has 26,000+ employees worldwide$$$$NVDA, Nvidia Corporation, nVidia)
("entity"$$$$Jensen Huang$$$$Person$$$$CEO of Nvidia who announced Blackwell B200 GPU$$$$Jensen, J. Huang)
("entity"$$$$Q3 2024$$$$Date$$$$Third quarter of fiscal year 2024, July-September 2024$$$$Q3 FY2024, 3Q24)
("entity"$$$$$35.1B$$$$Amount$$$$Nvidia's Q3 2024 revenue of 35.1 billion dollars$$$$35.1 billion, $35.1 billion)
("entity"$$$$94%$$$$Percentage$$$$Year-over-year revenue growth rate$$$$94 percent)
("entity"$$$$Blackwell B200$$$$Product$$$$New GPU product from Nvidia shipping in early 2025$$$$B200, Blackwell GPU, B200 GPU)
("entity"$$$$26,000$$$$Number$$$$Number of Nvidia employees worldwide$$$$26000, 26K)
("entity"$$$$early 2025$$$$Date$$$$Expected shipping date for Blackwell B200$$$$Q1 2025)
("relationship"$$$$Jensen Huang$$$$Nvidia$$$$is CEO of$$$$Chief Executive Officer$$$$none$$$$present)
("relationship"$$$$Nvidia$$$$$35.1B$$$$reported revenue of$$$$Q3 2024 quarterly revenue$$$$2024-07$$$$2024-09)
("relationship"$$$$Nvidia$$$$94%$$$$achieved growth of$$$$Year-over-year revenue growth in Q3 2024$$$$2024-07$$$$2024-09)
("relationship"$$$$Nvidia$$$$Blackwell B200$$$$is developing$$$$New GPU product$$$$none$$$$present)
("relationship"$$$$Blackwell B200$$$$early 2025$$$$ships in$$$$Expected shipping date$$$$2025-01$$$$2025-03)
("relationship"$$$$Nvidia$$$$26,000$$$$has employees$$$$Total employee count$$$$none$$$$present)
("relationship"$$$$Jensen Huang$$$$Blackwell B200$$$$announced$$$$CEO announced the product$$$$2024$$$$2024)

## YOUR TASK - BE EXHAUSTIVE!

Extract EVERYTHING from this text. Target {max_triplets}+ triplets. Include:
- Every named entity (people, companies, products, places)
- Every numeric value (amounts, percentages, counts)  
- Every date or time period
- Every relationship between any two entities
- Every attribute or property of any entity

Text to extract from:

{text}

## OUTPUT (entities first, then relationships) - Extract MORE than you think is needed!
"""


# Coreference Resolution Prompt - Links entities across documents
COREFERENCE_PROMPT = """
-Goal-
Identify which entities in Document B refer to the SAME entity as in Document A.
Link aliases, pronouns, and alternative names to their canonical entity.

-Document A Entities-
{entities_a}

-Document B Text-
{text_b}

-Instructions-
1. Find mentions in Document B that refer to entities from Document A
2. Consider: pronouns (he, she, they), titles (Dr., CEO), nicknames, abbreviations
3. "The company" might refer to an organization from Doc A
4. "He" or "She" might refer to a person from Doc A

-Format-
("coreference"$$$$<mention_in_doc_b>$$$$<canonical_entity_from_doc_a>$$$$<confidence_0_to_1>)

-Examples-
("coreference"$$$$The tech giant$$$$Apple Inc.$$$$0.9)
("coreference"$$$$He$$$$Tim Cook$$$$0.85)
("coreference"$$$$Dr. Smith$$$$John Smith$$$$0.95)
("coreference"$$$$The CEO$$$$Elon Musk$$$$0.8)

-Output (list all coreferences found)-
"""


@dataclass
class ExtractionConfig:
    """Configuration for knowledge extraction."""
    chunk_size: int = 2048
    chunk_overlap: int = 200
    max_triplets_per_chunk: int = 100  # Increased for exhaustive extraction
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
        self.config = ExtractionConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_triplets_per_chunk=max_triplets_per_chunk,
            max_workers=max_workers,
        )
        
        # Resolver factory (per-worker / per-call) to avoid shared-state contention
        from graphmem.graph.entity_resolver import EntityResolver
        if entity_resolver is None:
            # Default thresholds
            similarity_threshold = 0.85
            token_threshold = 0.7
            fuzzy_threshold = 0.92
        else:
            similarity_threshold = getattr(entity_resolver, "similarity_threshold", 0.85)
            token_threshold = getattr(entity_resolver, "token_threshold", 0.7)
            fuzzy_threshold = getattr(entity_resolver, "fuzzy_threshold", 0.92)
        
        def _factory():
            return EntityResolver(
                embeddings=self.embeddings,
                similarity_threshold=similarity_threshold,
                token_threshold=token_threshold,
                fuzzy_threshold=fuzzy_threshold,
            )
        
        self._resolver_factory = _factory
        # Keep a prototype reference for compatibility but don't reuse it in parallel
        self.entity_resolver = None
    
    def extract(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_id: str,
        user_id: str = "default",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        existing_nodes: Optional[List[MemoryNode]] = None,
        enable_coreference: bool = True,
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """
        Extract entities and relationships from content.
        
        Args:
            content: Text content to process
            metadata: Metadata to attach to extracted elements
            memory_id: ID of parent memory
            user_id: User ID for multi-tenant isolation
            progress_callback: Optional progress callback
            existing_nodes: Optional list of already-known entities for coreference resolution
            enable_coreference: Whether to run coreference resolution (default True)
        
        Returns:
            Tuple of (nodes, edges)
        
        NEW FEATURES:
        - Extracts entity ALIASES (nicknames, abbreviations, titles)
        - Extracts TEMPORAL VALIDITY (valid_from, valid_until) for relationships
        - Runs COREFERENCE RESOLUTION to link new mentions to existing entities
        """
        if not content or not content.strip():
            return [], []
        
        # FACT ORDER TRACKING: Parse from ORIGINAL content before chunking
        # This preserves newlines for proper fact number detection
        fact_priorities = self._parse_fact_order(content)
        if fact_priorities:
            logger.info(f"📊 Global fact priorities: {dict(list(fact_priorities.items())[:10])}")
        
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
                chunks[0], metadata, memory_id, user_id, fact_priorities
            )
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        else:
            # Multiple chunks - process in parallel
            completed = 0
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._extract_from_chunk, chunk, metadata, memory_id, user_id, fact_priorities
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
        
        # Resolve entity duplicates with a per-call resolver (no shared contention)
        resolver = self._resolver_factory()
        nodes = resolver.resolve(all_entities, memory_id, user_id)
        
        # NEW: Run coreference resolution if we have existing entities
        if enable_coreference and existing_nodes and len(existing_nodes) > 0:
            logger.info(f"Running coreference resolution against {len(existing_nodes)} existing entities...")
            
            # Run coreference on the full content
            coreferences = self.resolve_coreferences(
                existing_nodes=existing_nodes,
                new_chunk=content[:5000],  # Sample of content
                memory_id=memory_id,
                user_id=user_id,
            )
            
            if coreferences:
                logger.info(f"Found {len(coreferences)} coreferences, applying to nodes...")
                nodes = self.apply_coreferences(nodes, coreferences, existing_nodes)
        
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
        user_id: str = "default",
        fact_priorities: Dict[str, int] = None,  # GLOBAL priorities from original content
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Extract entities and relationships from a single chunk."""
        
        # Use global fact_priorities if provided, else parse from chunk (fallback)
        if fact_priorities is None:
            fact_priorities = self._parse_fact_order(chunk)
        
        prompt = EXTRACTION_PROMPT.format(
            max_triplets=self.config.max_triplets_per_chunk,
            text=chunk,
        )
        
        try:
            response = self.llm.complete(prompt)
            entities, relationships = self._parse_extraction_response(response)
            
            # LLM-BASED TEMPORAL REASONING for relationships without clear dates
            relationships = self._llm_extract_temporal_context(chunk, relationships)
            
            # Create nodes with embeddings and ALIASES
            nodes = []
            for entity_data in entities:
                # Handle both old format (3-tuple) and new format (4-tuple with aliases)
                if len(entity_data) >= 4:
                    name, entity_type, description, aliases = entity_data[:4]
                else:
                    name, entity_type, description = entity_data[:3]
                    aliases = set()
                
                # Generate embedding for the node (name + description)
                text_to_embed = f"{name}: {description}" if description else name
                try:
                    embedding = self.embeddings.embed_text(text_to_embed)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for {name}: {e}")
                    embedding = None
                
                # Store the FULL source chunk for context during answer generation
                # This is critical for not losing information from the original text
                source_chunks = [chunk]  # Will accumulate during entity resolution
                
                node = MemoryNode(
                    id="",  # Will be generated
                    name=name,
                    entity_type=entity_type,
                    description=description,
                    aliases=aliases if isinstance(aliases, set) else set(aliases) if aliases else set(),
                    embedding=embedding,  # Add embedding for vector search
                    properties={
                        **metadata,
                        "source_chunk": chunk,  # FULL chunk, not truncated
                        "source_chunks": source_chunks,  # List of all chunks mentioning this entity
                        "extraction_context": chunk[:500],  # Preview for debugging
                    },
                    user_id=user_id,     # Multi-tenant isolation
                    memory_id=memory_id,
                )
                nodes.append(node)
            
            # Create edges with TEMPORAL VALIDITY
            edges = []
            for rel_data in relationships:
                # Handle both old format (4-tuple) and new format (6-tuple with temporal)
                if len(rel_data) >= 6:
                    source, target, relation, description, valid_from, valid_until = rel_data[:6]
                elif len(rel_data) >= 4:
                    source, target, relation, description = rel_data[:4]
                    valid_from, valid_until = None, None
                else:
                    continue
                
                # Parse temporal strings to datetime if possible
                from datetime import datetime
                valid_from_dt = self._parse_temporal_to_datetime(valid_from)
                valid_until_dt = self._parse_temporal_to_datetime(valid_until)
                
                # Get priority from fact order (higher = newer, should override)
                priority = self._get_fact_priority(source, target, fact_priorities)
                logger.debug(f"📊 Edge priority: {source} -> {target} = {priority}")
                
                # Map priority to importance: Higher fact number = higher importance
                from graphmem.core.memory_types import MemoryImportance
                if priority >= 100:
                    edge_importance = MemoryImportance.CRITICAL
                elif priority >= 50:
                    edge_importance = MemoryImportance.VERY_HIGH
                elif priority >= 20:
                    edge_importance = MemoryImportance.HIGH
                else:
                    edge_importance = MemoryImportance.MEDIUM
                
                edge = MemoryEdge(
                    id="",  # Will be generated
                    source_id=source,  # Will be resolved to canonical
                    target_id=target,
                    relation_type=relation,
                    description=description,
                    valid_from=valid_from_dt,
                    valid_until=valid_until_dt,
                    importance=edge_importance,  # PRIORITY-BASED IMPORTANCE
                    properties={
                        **metadata,
                        "temporal_raw": {"from": valid_from, "until": valid_until},
                        "priority": priority,  # Higher = newer info, overrides conflicts
                    },
                    memory_id=memory_id,
                )
                edges.append(edge)
            
            return nodes, edges
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk: {e}")
            if self.config.retry_on_failure:
                # Retry with simpler prompt
                return self._extract_fallback(chunk, metadata, memory_id, user_id)
            return [], []
    
    def _parse_fact_order(self, chunk: str) -> Dict[str, int]:
        """
        Parse fact numbers from numbered lists like:
        "43. Christianity was founded in Taipei"
        
        Returns dict mapping entity names to their fact number (priority).
        Higher number = newer information = should override.
        """
        fact_priorities = {}
        
        # Pattern: "123. Entity name..." or "123: Entity name..."
        for match in re.finditer(r'^(\d+)[.\s:]+(.+)$', chunk, re.MULTILINE):
            fact_num = int(match.group(1))
            fact_text = match.group(2).strip()
            
            # Extract individual capitalized entities (proper nouns)
            # Pattern: One or more capitalized words (entity names)
            for entity_match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', fact_text):
                entity = entity_match.group(1).strip()
                if len(entity) > 2 and entity.lower() not in ['the', 'and', 'for']:
                    # Update only if this is a higher fact number
                    existing = fact_priorities.get(entity.lower(), 0)
                    fact_priorities[entity.lower()] = max(existing, fact_num)
        
        return fact_priorities
    
    def _get_fact_priority(
        self,
        source: str,
        target: str,
        fact_priorities: Dict[str, int],
    ) -> int:
        """
        Get priority for an edge based on fact order.
        Higher priority = newer information = overrides conflicts.
        
        STRATEGY: Use TARGET entity's priority since it's more specific.
        E.g., "Christianity → founded_in → Taipei" uses Taipei's priority (43)
              "Christianity → founded_in → Jerusalem" uses Jerusalem's priority (40)
        """
        target_lower = target.lower()
        
        # Direct match for target
        target_priority = fact_priorities.get(target_lower, 0)
        
        # Partial match for target
        for entity, priority in fact_priorities.items():
            if entity in target_lower or target_lower in entity:
                if priority > target_priority:
                    target_priority = priority
        
        # If target has priority, use it (more specific)
        if target_priority > 0:
            return target_priority
        
        # Fallback to source priority
        source_lower = source.lower()
        source_priority = fact_priorities.get(source_lower, 0)
        for entity, priority in fact_priorities.items():
            if entity in source_lower or source_lower in entity:
                if priority > source_priority:
                    source_priority = priority
        
        return source_priority
    
    def _extract_fallback(
        self,
        chunk: str,
        metadata: Dict[str, Any],
        memory_id: str,
        user_id: str = "default",
    ) -> Tuple[List[MemoryNode], List[MemoryEdge]]:
        """Fallback extraction with simpler but still exhaustive prompt."""
        try:
            prompt = f"""Extract EVERYTHING from this text - be exhaustive!

EXTRACT:
- EVERY person, organization, product, location, date, number, percentage
- EVERY relationship between any two things
- ALL numeric values (revenue, employees, percentages, amounts)
- ALL dates and time periods
- ALL attributes and properties of entities
            
Text: {chunk}

Format your response EXACTLY like this:
ENTITY: name | type | detailed description with all facts
RELATIONSHIP: source -> relation -> target | description

Example:
ENTITY: Nvidia | Company | Semiconductor company, $35.1B Q3 2024 revenue, 94% YoY growth, 26K employees
ENTITY: $35.1B | Amount | Nvidia Q3 2024 revenue
ENTITY: Jensen Huang | Person | CEO of Nvidia
RELATIONSHIP: Nvidia -> has_revenue -> $35.1B | Q3 2024 revenue
RELATIONSHIP: Jensen Huang -> is CEO of -> Nvidia | Current CEO

Extract EVERYTHING - numbers, dates, amounts, percentages, all relationships:"""
            
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
                    properties={
                        **metadata,
                        "source_chunk": chunk,  # FULL chunk for context
                        "source_chunks": [chunk],  # Accumulates during resolution
                        "extraction_context": chunk[:500],
                    }, 
                    user_id=user_id,     # Multi-tenant isolation
                    memory_id=memory_id,
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
        """
        Parse LLM extraction response with aliases and temporal validity.
        
        New format:
        - Entity: (name, type, description, aliases)
        - Relationship: (source, target, relation, description, valid_from, valid_until)
        """
        entities = []
        relationships = []
        
        # Line-based parsing (most reliable)
        for line in response.split('\n'):
            line = line.strip()
            if not line or '$$$$' not in line:
                continue
            
            # Parse entity with aliases
            if 'entity' in line.lower():
                parts = [p.strip().strip('"').strip("'").strip(')').strip('(') for p in line.split('$$$$')]
                if len(parts) >= 4:
                    name = parts[1] if len(parts) > 1 else ""
                    entity_type = parts[2] if len(parts) > 2 else "unknown"
                    description = parts[3] if len(parts) > 3 else ""
                    # NEW: Parse aliases (5th field)
                    aliases_str = parts[4] if len(parts) > 4 else "none"
                    aliases = self._parse_aliases(aliases_str)
                    
                    if name:
                        entities.append((name, entity_type, description, aliases))
            
            # Parse relationship with temporal validity
            elif 'relationship' in line.lower():
                parts = [p.strip().strip('"').strip("'").strip(')').strip('(') for p in line.split('$$$$')]
                if len(parts) >= 5:
                    source = parts[1] if len(parts) > 1 else ""
                    target = parts[2] if len(parts) > 2 else ""
                    relation = parts[3] if len(parts) > 3 else ""
                    description = parts[4] if len(parts) > 4 else ""
                    # NEW: Parse temporal validity (6th and 7th fields)
                    valid_from = self._parse_temporal(parts[5]) if len(parts) > 5 else None
                    valid_until = self._parse_temporal(parts[6]) if len(parts) > 6 else None
                    
                    if source and target and relation:
                        relationships.append((source, target, relation, description, valid_from, valid_until))
        
        logger.debug(f"Parsed {len(entities)} entities (with aliases), {len(relationships)} relationships (with temporal)")
        return entities, relationships
    
    def _parse_aliases(self, aliases_str: str) -> set:
        """Parse comma-separated aliases into a set."""
        if not aliases_str or aliases_str.lower() in ('none', 'n/a', '-', ''):
            return set()
        
        aliases = set()
        for alias in aliases_str.split(','):
            alias = alias.strip()
            if alias and alias.lower() not in ('none', 'n/a'):
                aliases.add(alias)
        
        return aliases
    
    def _parse_temporal(self, temporal_str: str) -> Optional[str]:
        """Parse temporal string (date, year, or 'present')."""
        if not temporal_str:
            return None
        
        temporal_str = temporal_str.strip().lower()
        
        if temporal_str in ('none', 'n/a', '-', '', 'null'):
            return None
        
        if temporal_str == 'present':
            return 'present'
        
        # Try to extract year or date
        # Match YYYY-MM-DD or YYYY
        date_match = re.search(r'(\d{4}(?:-\d{2}-\d{2})?)', temporal_str)
        if date_match:
            return date_match.group(1)
        
        return temporal_str  # Return as-is if can't parse
    
    def _llm_extract_temporal_context(
        self,
        text: str,
        relationships: List[Tuple],
    ) -> List[Tuple]:
        """
        LLM-BASED TEMPORAL REASONING
        
        Ask the LLM to reason about temporal validity for each relationship,
        especially when dates aren't explicitly stated.
        """
        if not self.llm or not relationships:
            return relationships
        
        # Find relationships without clear temporal data
        needs_temporal = []
        for i, rel in enumerate(relationships):
            valid_from = rel[4] if len(rel) > 4 else None
            valid_until = rel[5] if len(rel) > 5 else None
            if not valid_from and not valid_until:
                needs_temporal.append((i, rel))
        
        if not needs_temporal:
            return relationships
        
        # Build prompt for LLM temporal reasoning
        rel_descriptions = []
        for idx, (i, rel) in enumerate(needs_temporal):
            rel_descriptions.append(
                f"{idx+1}. {rel[0]} --[{rel[2]}]--> {rel[1]}: {rel[3] if len(rel) > 3 else 'no description'}"
            )
        
        prompt = f"""Analyze the temporal validity of these relationships based on the source text.

SOURCE TEXT:
{text[:2000]}

RELATIONSHIPS TO ANALYZE:
{chr(10).join(rel_descriptions)}

For EACH relationship, determine:
1. Is this a CURRENT/ONGOING relationship or a PAST relationship?
2. When did it START? (year or "unknown")
3. When did it END? ("present" if ongoing, year if ended, "unknown" if unclear)

Look for clues like:
- "was CEO" vs "is CEO" (past vs current)
- "since 2020" (start date)
- "from 2015 to 2018" (start and end)
- "currently" or "now" (present)
- Past tense verbs suggest ended relationships

FORMAT (one line per relationship):
<relationship_number>|<valid_from>|<valid_until>

Example:
1|2015|present
2|2010|2018
3|unknown|unknown"""

        try:
            response = self.llm.complete(prompt)
            
            # Parse response and update relationships
            updated = list(relationships)
            for line in response.strip().split('\n'):
                if '|' not in line:
                    continue
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    try:
                        rel_num = int(parts[0].strip()) - 1
                        valid_from = parts[1].strip()
                        valid_until = parts[2].strip()
                        
                        if 0 <= rel_num < len(needs_temporal):
                            orig_idx, orig_rel = needs_temporal[rel_num]
                            # Update the relationship with temporal data
                            new_rel = list(orig_rel)
                            while len(new_rel) < 6:
                                new_rel.append(None)
                            new_rel[4] = self._parse_temporal(valid_from)
                            new_rel[5] = self._parse_temporal(valid_until)
                            updated[orig_idx] = tuple(new_rel)
                            logger.debug(f"LLM temporal: {orig_rel[0]} -> {orig_rel[1]}: {valid_from} to {valid_until}")
                    except (ValueError, IndexError):
                        continue
            
            return updated
            
        except Exception as e:
            logger.warning(f"LLM temporal reasoning failed: {e}")
            return relationships
    
    def _parse_temporal_to_datetime(self, temporal_str: Optional[str]):
        """Convert temporal string to datetime object."""
        from datetime import datetime
        
        if not temporal_str:
            return None
        
        temporal_str = str(temporal_str).strip().lower()
        
        if temporal_str in ('none', 'n/a', '-', '', 'null', 'present'):
            return None  # 'present' means ongoing, so valid_until is None
        
        try:
            # Try YYYY-MM-DD
            if '-' in temporal_str and len(temporal_str) == 10:
                return datetime.strptime(temporal_str, '%Y-%m-%d')
            
            # Try YYYY
            if len(temporal_str) == 4 and temporal_str.isdigit():
                return datetime.strptime(temporal_str, '%Y')
            
            # Try to find a year
            year_match = re.search(r'(\d{4})', temporal_str)
            if year_match:
                return datetime.strptime(year_match.group(1), '%Y')
            
        except ValueError:
            pass
        
        return None
    
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
    
    def resolve_coreferences(
        self,
        existing_nodes: List[MemoryNode],
        new_chunk: str,
        memory_id: str,
        user_id: str = "default",
    ) -> Dict[str, str]:
        """
        Resolve coreferences between existing entities and mentions in new text.
        
        Uses LLM to identify which mentions in new_chunk refer to existing entities.
        Returns a mapping of {mention_in_chunk: canonical_entity_name}
        
        This enables cross-document entity linking for:
        - Pronouns: "He" → "Elon Musk"
        - Titles: "The CEO" → "Tim Cook"
        - Nicknames: "The Iron Man of Tech" → "Elon Musk"
        - Abbreviations: "Dr. Chen" → "Alexander Chen"
        """
        if not existing_nodes or not new_chunk:
            return {}
        
        # Format existing entities for the prompt
        entities_str = "\n".join([
            f"- {node.name} ({node.entity_type}): {node.description or 'No description'}"
            for node in existing_nodes[:50]  # Limit to avoid token overflow
        ])
        
        prompt = COREFERENCE_PROMPT.format(
            entities_a=entities_str,
            text_b=new_chunk[:3000],  # Limit chunk size
        )
        
        try:
            response = self.llm.complete(prompt)
            return self._parse_coreference_response(response)
        except Exception as e:
            logger.warning(f"Coreference resolution failed: {e}")
            return {}
    
    def _parse_coreference_response(self, response: str) -> Dict[str, str]:
        """Parse coreference response into mention→canonical mapping."""
        import re
        coreferences = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if 'coreference' in line.lower() and '$$$$' in line:
                parts = [p.strip().strip('"').strip("'").strip(')').strip('(') for p in line.split('$$$$')]
                if len(parts) >= 3:
                    mention = parts[1]
                    canonical = parts[2]
                    
                    # Parse confidence - extract just the numeric part
                    confidence = 0.5
                    if len(parts) > 3:
                        conf_str = parts[3]
                        # Extract just the decimal number (e.g., "0.95)," -> "0.95")
                        match = re.search(r'(\d+\.?\d*)', conf_str)
                        if match:
                            try:
                                confidence = float(match.group(1))
                            except ValueError:
                                confidence = 0.5
                    
                    # Only accept high-confidence coreferences
                    if confidence >= 0.7 and mention and canonical:
                        coreferences[mention.lower()] = canonical
        
        logger.info(f"Resolved {len(coreferences)} coreferences")
        return coreferences
    
    def apply_coreferences(
        self,
        nodes: List[MemoryNode],
        coreferences: Dict[str, str],
        existing_nodes: List[MemoryNode],
    ) -> List[MemoryNode]:
        """
        Apply coreference mappings to link new nodes to existing canonical entities.
        
        If a new node's name matches a coreference mention, add the canonical
        entity name as an alias and update the canonical_name field.
        """
        if not coreferences:
            return nodes
        
        # Build canonical name lookup
        canonical_lookup = {node.name.lower(): node.name for node in existing_nodes}
        
        updated_nodes = []
        for node in nodes:
            node_name_lower = node.name.lower()
            
            # Check if this node's name is a coreference mention
            if node_name_lower in coreferences:
                canonical = coreferences[node_name_lower]
                
                # Update node to link to canonical entity
                new_aliases = node.aliases.copy() if node.aliases else set()
                new_aliases.add(node.name)  # Add original name as alias
                
                # Find the actual canonical entity
                canonical_actual = canonical_lookup.get(canonical.lower(), canonical)
                
                updated_node = MemoryNode(
                    id=node.id,
                    name=canonical_actual,  # Use canonical name
                    entity_type=node.entity_type,
                    description=node.description,
                    canonical_name=canonical_actual,
                    aliases=new_aliases,
                    embedding=node.embedding,
                    properties={**node.properties, "coreference_resolved": True, "original_mention": node.name},
                    user_id=node.user_id,
                    memory_id=node.memory_id,
                )
                updated_nodes.append(updated_node)
                logger.debug(f"Resolved coreference: '{node.name}' → '{canonical_actual}'")
            else:
                updated_nodes.append(node)
        
        return updated_nodes


"""
GraphMem Entity Resolver

Resolves and merges duplicate entities using semantic similarity,
token matching, and embedding-based comparison.

This is critical for production-grade memory systems where the same
entity may be mentioned in different ways across documents.
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import numpy as np

from graphmem.core.memory_types import MemoryNode

logger = logging.getLogger(__name__)


# Common stopwords to ignore in entity names
ENTITY_STOPWORDS = {
    "the", "a", "an", "of", "and", "&", "inc", "ltd", "llc", "plc",
    "corp", "co", "company", "limited", "sa", "s.a.", "gmbh", "ag",
    "corporation", "incorporated", "enterprises", "holdings", "group",
}


@dataclass
class EntityCandidate:
    """A candidate entity with metadata for resolution."""
    name: str
    entity_type: str
    description: str
    tokens: Set[str] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    aliases: Set[str] = field(default_factory=set)
    descriptions: Set[str] = field(default_factory=set)
    occurrences: int = 1
    original_node: Optional[MemoryNode] = None


class EntityResolver:
    """
    Resolves duplicate entities using multiple matching strategies.
    
    Strategies:
    1. Exact match - Same name after normalization
    2. Token match - High overlap in name tokens
    3. Semantic match - High embedding similarity
    4. Fuzzy match - High string similarity ratio
    
    The resolver maintains an index of canonical entities and maps
    all variants to them.
    """
    
    def __init__(
        self,
        embeddings,
        similarity_threshold: float = 0.85,
        token_threshold: float = 0.7,
        fuzzy_threshold: float = 0.92,
    ):
        """
        Initialize entity resolver.
        
        Args:
            embeddings: Embedding provider for semantic matching
            similarity_threshold: Threshold for embedding similarity
            token_threshold: Threshold for token overlap
            fuzzy_threshold: Threshold for fuzzy string matching
        """
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.token_threshold = token_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
        # Canonical entity index
        self._entity_index: Dict[str, EntityCandidate] = {}
        self._alias_lookup: Dict[str, str] = {}  # alias -> canonical key
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def resolve(
        self,
        nodes: List[MemoryNode],
        memory_id: str,
        user_id: str = "default",
    ) -> List[MemoryNode]:
        """
        Resolve duplicate entities in a list of nodes.
        
        Args:
            nodes: List of nodes to resolve
            memory_id: ID of parent memory
            user_id: User ID for multi-tenant isolation
        
        Returns:
            List of deduplicated nodes with canonical names
        """
        if not nodes:
            return []
        
        # Fast, per-instance resolution (resolver instances are not shared across threads)
        local_index = self._entity_index
        local_alias = self._alias_lookup
        resolved_nodes: List[MemoryNode] = []
        
        for node in nodes:
            canonical, key = self._resolve_entity_from_maps(
                node=node,
                index_map=local_index,
                alias_map=local_alias,
            )
            
            if canonical:
                # Merge with existing canonical in local map
                merged = self._merge_entities(canonical, node)
                local_index[key] = merged
                
                best_embedding = merged.embedding
                if best_embedding is None and node.embedding is not None:
                    best_embedding = np.array(node.embedding) if isinstance(node.embedding, list) else node.embedding
                
                # Get accumulated source chunks from original node
                source_chunks = []
                if merged.original_node and hasattr(merged.original_node, 'properties'):
                    source_chunks = merged.original_node.properties.get('source_chunks', [])
                
                # Also include new node's source chunks
                if node.properties:
                    new_chunks = node.properties.get('source_chunks', [])
                    for chunk in new_chunks:
                        if chunk and chunk not in source_chunks:
                            source_chunks.append(chunk)
                    new_chunk = node.properties.get('source_chunk', '')
                    if new_chunk and new_chunk not in source_chunks:
                        source_chunks.append(new_chunk)
                
                resolved_node = MemoryNode(
                    id=key,  # Use canonical key as ID
                    name=merged.name,
                    entity_type=merged.entity_type or node.entity_type,
                    description=self._best_description(merged.descriptions),
                    canonical_name=merged.name,
                    aliases=merged.aliases,
                    embedding=list(best_embedding) if best_embedding is not None else None,  # Preserve embedding!
                    properties={
                        **node.properties,
                        "canonical_name": merged.name,
                        "aliases": list(merged.aliases),
                        "occurrence_count": merged.occurrences,
                        "source_chunks": source_chunks,  # ACCUMULATED chunks from all mentions
                    },
                    user_id=user_id,  # Multi-tenant isolation
                    memory_id=memory_id,
                )
                
                if not any(n.id == resolved_node.id for n in resolved_nodes):
                    resolved_nodes.append(resolved_node)
            else:
                # New entity: add to local maps
                candidate = self._create_candidate(node)
                key = self._generate_key(node.name)
                local_index[key] = candidate
                
                # CRITICAL: Register ALL aliases from node.aliases + the name itself!
                self._register_alias(node.name, key)
                if hasattr(node, 'aliases') and node.aliases:
                    for alias in node.aliases:
                        self._register_alias(alias, key)
                        # Also register in local alias map for current batch
                        local_alias[alias.lower()] = key
                        slug = re.sub(r'[^a-z0-9]', '', alias.lower())
                        if slug:
                            local_alias.setdefault(slug, key)
                
                # Collect all aliases for the node
                all_aliases = {node.name}
                if hasattr(node, 'aliases') and node.aliases:
                    all_aliases.update(node.aliases)
                
                resolved_node = MemoryNode(
                    id=key,
                    name=node.name,
                    entity_type=node.entity_type,
                    description=node.description,
                    canonical_name=node.name,
                    aliases=all_aliases,  # Include ALL LLM-extracted aliases!
                    embedding=node.embedding,  # Preserve embedding!
                    properties={
                        **node.properties,
                        "canonical_name": node.name,
                        "aliases": list(all_aliases),
                    },
                    user_id=user_id,  # Multi-tenant isolation
                    memory_id=memory_id,
                )
                resolved_nodes.append(resolved_node)
        
        logger.info(f"Resolved {len(nodes)} nodes to {len(resolved_nodes)} unique entities")
        return resolved_nodes
    
    def _resolve_entity(
        self,
        node: MemoryNode,
    ) -> Tuple[Optional[EntityCandidate], Optional[str]]:
        """
        Try to resolve a node to an existing canonical entity.
        
        Returns tuple of (canonical_candidate, canonical_key) or (None, None).
        """
        cleaned_name = self._clean_name(node.name)
        tokens = self._tokenize(cleaned_name)
        
        # Check direct alias lookup
        direct_key = self._alias_lookup.get(cleaned_name.lower())
        if direct_key and direct_key in self._entity_index:
            return self._entity_index[direct_key], direct_key
        
        # Check slug alias
        slug = re.sub(r'[^a-z0-9]', '', cleaned_name.lower())
        slug_key = self._alias_lookup.get(slug)
        if slug_key and slug_key in self._entity_index:
            return self._entity_index[slug_key], slug_key
        
        # Get embedding for semantic comparison
        embedding = self._get_embedding(node.description or cleaned_name)
        
        best_match = None
        best_score = 0.0
        best_key = None
        
        for key, candidate in self._entity_index.items():
            # Skip if entity types don't match (when both are specified)
            if (
                node.entity_type and
                candidate.entity_type and
                node.entity_type.lower() != candidate.entity_type.lower()
            ):
                continue
            
            # Calculate match scores
            token_score = self._token_similarity(tokens, candidate.tokens)
            fuzzy_score = self._fuzzy_similarity(cleaned_name, candidate.name)
            embedding_score = self._embedding_similarity(embedding, candidate.embedding)
            
            # Check if it qualifies as a match
            qualifies = (
                fuzzy_score >= self.fuzzy_threshold or
                token_score >= self.token_threshold or
                (token_score >= 0.6 and embedding_score >= self.similarity_threshold) or
                (embedding_score >= 0.92 and fuzzy_score >= 0.85)
            )
            
            if not qualifies:
                continue
            
            # Calculate combined score
            combined = max(
                fuzzy_score,
                token_score,
                (token_score * 0.4 + embedding_score * 0.6) if embedding_score else 0.0,
            )
            
            if combined > best_score:
                best_score = combined
                best_match = candidate
                best_key = key
        
        if best_match:
            logger.debug(f"Resolved '{node.name}' to '{best_match.name}' (score={best_score:.2f})")
            return best_match, best_key
        
        return None, None

    # ---------- Resolution helper using provided maps ----------
    def _resolve_entity_from_maps(
        self,
        node: MemoryNode,
        index_map: Dict[str, EntityCandidate],
        alias_map: Dict[str, str],
    ) -> Tuple[Optional[EntityCandidate], Optional[str]]:
        """Resolve using provided maps."""
        cleaned_name = self._clean_name(node.name)
        tokens = self._tokenize(cleaned_name)
        
        # Check direct alias lookup
        direct_key = alias_map.get(cleaned_name.lower())
        if direct_key and direct_key in index_map:
            return index_map[direct_key], direct_key
        
        # Check slug alias
        slug = re.sub(r'[^a-z0-9]', '', cleaned_name.lower())
        slug_key = alias_map.get(slug)
        if slug_key and slug_key in index_map:
            return index_map[slug_key], slug_key
        
        # Get embedding for semantic comparison (cache is per-resolver instance)
        embedding = self._get_embedding(node.description or cleaned_name)
        
        best_match = None
        best_score = 0.0
        best_key = None
        
        for key, candidate in index_map.items():
            if (
                node.entity_type and
                candidate.entity_type and
                node.entity_type.lower() != candidate.entity_type.lower()
            ):
                continue
            
            token_score = self._token_similarity(tokens, candidate.tokens)
            fuzzy_score = self._fuzzy_similarity(cleaned_name, candidate.name)
            embedding_score = self._embedding_similarity(embedding, candidate.embedding)
            
            qualifies = (
                fuzzy_score >= self.fuzzy_threshold or
                token_score >= self.token_threshold or
                (token_score >= 0.6 and embedding_score >= self.similarity_threshold) or
                (embedding_score >= 0.92 and fuzzy_score >= 0.85)
            )
            if not qualifies:
                continue
            
            combined = max(
                fuzzy_score,
                token_score,
                (token_score * 0.4 + embedding_score * 0.6) if embedding_score else 0.0,
            )
            
            if combined > best_score:
                best_score = combined
                best_match = candidate
                best_key = key
        
        if best_match:
            return best_match, best_key
        
        return None, None
    
    
    def _create_candidate(self, node: MemoryNode) -> EntityCandidate:
        """Create an entity candidate from a node, including LLM-extracted aliases."""
        cleaned_name = self._clean_name(node.name)
        embedding = self._get_embedding(node.description or cleaned_name)
        
        # CRITICAL: Include LLM-extracted aliases from node.aliases!
        aliases = {node.name, cleaned_name}
        if hasattr(node, 'aliases') and node.aliases:
            aliases.update(node.aliases)
        
        # Also add all cleaned versions of aliases
        for alias in list(aliases):
            cleaned = self._clean_name(alias)
            if cleaned:
                aliases.add(cleaned)
        
        # Build tokens from all aliases
        all_tokens = self._tokenize(cleaned_name)
        for alias in aliases:
            all_tokens.update(self._tokenize(alias))
        
        return EntityCandidate(
            name=node.name,
            entity_type=node.entity_type,
            description=node.description or "",
            tokens=all_tokens,
            embedding=embedding,
            aliases=aliases,
            descriptions={node.description} if node.description else set(),
            occurrences=1,
            original_node=node,
        )
    
    def _merge_entities(
        self,
        canonical: EntityCandidate,
        new_node: MemoryNode,
    ) -> EntityCandidate:
        """Merge a new node into an existing canonical entity, including aliases and source chunks."""
        cleaned_name = self._clean_name(new_node.name)
        
        # Update aliases from node name
        canonical.aliases.add(new_node.name)
        canonical.aliases.add(cleaned_name)
        
        # CRITICAL: Also merge LLM-extracted aliases from new_node.aliases!
        if hasattr(new_node, 'aliases') and new_node.aliases:
            for alias in new_node.aliases:
                canonical.aliases.add(alias)
                cleaned = self._clean_name(alias)
                if cleaned:
                    canonical.aliases.add(cleaned)
        
        # Update tokens from ALL aliases
        canonical.tokens.update(self._tokenize(cleaned_name))
        for alias in canonical.aliases:
            canonical.tokens.update(self._tokenize(alias))
        
        # Update descriptions
        if new_node.description:
            canonical.descriptions.add(new_node.description)
        
        # ACCUMULATE SOURCE CHUNKS - critical for context during answer generation
        if hasattr(new_node, 'properties') and new_node.properties:
            new_chunks = new_node.properties.get('source_chunks', [])
            new_chunk = new_node.properties.get('source_chunk', '')
            
            # Get existing chunks from original node
            existing_chunks = []
            if canonical.original_node and hasattr(canonical.original_node, 'properties'):
                existing_chunks = canonical.original_node.properties.get('source_chunks', [])
            
            # Merge chunks (avoid duplicates)
            all_chunks = list(existing_chunks)
            for chunk in new_chunks:
                if chunk and chunk not in all_chunks:
                    all_chunks.append(chunk)
            if new_chunk and new_chunk not in all_chunks:
                all_chunks.append(new_chunk)
            
            # Store accumulated chunks back on the original node's properties
            if canonical.original_node:
                if not hasattr(canonical.original_node, 'properties') or canonical.original_node.properties is None:
                    canonical.original_node.properties = {}
                canonical.original_node.properties['source_chunks'] = all_chunks
        
        # Blend embeddings
        if new_node.description:
            new_embedding = self._get_embedding(new_node.description)
            if new_embedding is not None and canonical.embedding is not None:
                # Weighted average favoring newer information
                weight = canonical.occurrences / (canonical.occurrences + 1)
                canonical.embedding = (
                    canonical.embedding * weight +
                    new_embedding * (1 - weight)
                )
        
        # Choose best display name (prefer longer, more complete names)
        canonical.name = self._choose_display_name(canonical.name, new_node.name)
        
        # Increment occurrences
        canonical.occurrences += 1
        
        # Register new aliases
        key = self._generate_key(canonical.name)
        self._register_alias(new_node.name, key)
        self._register_alias(cleaned_name, key)
        
        return canonical
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize an entity name."""
        if not name:
            return ""
        cleaned = re.sub(r'\s+', ' ', str(name)).strip()
        cleaned = cleaned.strip(' "\'""\'\'')
        return cleaned
    
    def _tokenize(self, name: str) -> Set[str]:
        """Tokenize an entity name."""
        if not name:
            return set()
        tokens = set(re.findall(r'[a-z0-9]+', name.lower()))
        return tokens - ENTITY_STOPWORDS
    
    def _generate_key(self, name: str) -> str:
        """Generate a canonical key for an entity."""
        cleaned = self._clean_name(name).lower()
        return re.sub(r'[^a-z0-9]', '_', cleaned)[:64]
    
    def _register_alias(self, alias: str, canonical_key: str) -> None:
        """Register an alias pointing to a canonical key."""
        if not alias:
            return
        lowered = alias.lower()
        self._alias_lookup[lowered] = canonical_key
        
        slug = re.sub(r'[^a-z0-9]', '', lowered)
        if slug:
            self._alias_lookup.setdefault(slug, canonical_key)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if not text or not text.strip():
            return None
        
        cache_key = text[:500].lower()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            embedding = self.embeddings.embed_text(text)
            if embedding:
                vector = np.array(embedding, dtype=np.float32)
                self._embedding_cache[cache_key] = vector
                return vector
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
        
        return None
    
    def _token_similarity(self, tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Calculate Jaccard similarity between token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        
        if union == 0:
            return 0.0
        
        score = intersection / union
        
        # Boost if one is subset of another
        if tokens_a <= tokens_b or tokens_b <= tokens_a:
            score = max(score, 0.9)
        
        # Boost if they share the same "key" token (usually last token)
        sorted_a = sorted(tokens_a)
        sorted_b = sorted(tokens_b)
        if sorted_a and sorted_b and sorted_a[-1] == sorted_b[-1]:
            score = max(score, 0.7)
        
        return score
    
    def _fuzzy_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate fuzzy string similarity."""
        if not name_a or not name_b:
            return 0.0
        return SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
    
    def _embedding_similarity(
        self,
        embedding_a: Optional[np.ndarray],
        embedding_b: Optional[np.ndarray],
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        if embedding_a is None or embedding_b is None:
            return 0.0
        
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))
    
    def _choose_display_name(self, current: str, candidate: str) -> str:
        """Choose the best display name between two options."""
        if not current:
            return candidate
        if not candidate:
            return current
        
        current_tokens = len(current.split())
        candidate_tokens = len(candidate.split())
        
        # Prefer longer names (more complete)
        if candidate_tokens > current_tokens:
            return candidate
        
        # Same token count - prefer longer string
        if candidate_tokens == current_tokens and len(candidate) > len(current):
            return candidate
        
        # Same token count - prefer one that starts with the other
        if candidate_tokens == current_tokens and candidate.lower().startswith(current.lower()):
            return candidate
        
        return current
    
    def _best_description(self, descriptions: Set[str]) -> str:
        """Choose the best description from a set."""
        if not descriptions:
            return ""
        
        # Prefer longest description (most complete)
        return max(descriptions, key=len)
    
    def clear(self) -> None:
        """Clear the entity index."""
        self._entity_index.clear()
        self._alias_lookup.clear()
        self._embedding_cache.clear()


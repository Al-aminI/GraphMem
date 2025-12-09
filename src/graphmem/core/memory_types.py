"""
GraphMem Memory Types

Core data structures for the memory system.
Designed for production with serialization, validation, and immutability where appropriate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Set, Tuple, Union
from uuid import uuid4
import json
import hashlib


class MemoryImportance(Enum):
    """
    Memory importance levels for prioritization and decay.
    
    Higher importance memories are retained longer and prioritized in retrieval.
    """
    CRITICAL = 10      # Never decay, always retrieve first
    VERY_HIGH = 8      # Extremely slow decay, high priority
    HIGH = 6           # Slow decay, above average priority
    MEDIUM = 5         # Normal decay rate, average priority
    LOW = 3            # Faster decay, below average priority
    VERY_LOW = 1       # Fast decay, low priority
    EPHEMERAL = 0      # Immediate decay candidate
    
    @classmethod
    def from_score(cls, score: float) -> "MemoryImportance":
        """Convert a 0-10 score to importance level."""
        if score >= 9:
            return cls.CRITICAL
        elif score >= 7:
            return cls.VERY_HIGH
        elif score >= 5.5:
            return cls.HIGH
        elif score >= 4:
            return cls.MEDIUM
        elif score >= 2:
            return cls.LOW
        elif score >= 0.5:
            return cls.VERY_LOW
        return cls.EPHEMERAL


class MemoryState(Enum):
    """Memory lifecycle states."""
    ACTIVE = auto()           # Normal, accessible memory
    CONSOLIDATING = auto()    # Being merged with other memories
    DECAYING = auto()         # Marked for decay, still accessible
    ARCHIVED = auto()         # Moved to long-term storage
    DELETED = auto()          # Soft-deleted, pending cleanup


class EvolutionType(Enum):
    """Types of memory evolution events."""
    CONSOLIDATION = auto()    # Multiple memories merged
    REINFORCEMENT = auto()    # Memory strengthened by access
    DECAY = auto()            # Memory weakened over time
    REHYDRATION = auto()      # Memory restored from archive
    CORRECTION = auto()       # Memory updated with new info
    SYNTHESIS = auto()        # New memory created from existing
    PRUNING = auto()          # Low-value memory removed


@dataclass
class MemoryNode:
    """
    A node in the memory graph representing an entity.
    
    Nodes are immutable after creation to ensure graph integrity.
    Use MemoryNode.evolve() to create updated versions.
    
    Multi-Tenant Isolation:
        - user_id: Identifies the user/tenant (required for isolation)
        - memory_id: Identifies the specific memory session within a user
    """
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    canonical_name: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    importance: MemoryImportance = MemoryImportance.MEDIUM
    state: MemoryState = MemoryState.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    user_id: Optional[str] = None     # User/tenant isolation
    memory_id: Optional[str] = None   # Memory session reference
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.canonical_name:
            self.canonical_name = self.name
        if self.name:
            self.aliases.add(self.name)
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from name and type."""
        content = f"{self.name}:{self.entity_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def evolve(self, **updates) -> "MemoryNode":
        """Create a new node with updates, preserving history."""
        new_props = {**self.__dict__}
        new_props.update(updates)
        new_props["updated_at"] = datetime.utcnow()
        return MemoryNode(**new_props)
    
    def record_access(self) -> "MemoryNode":
        """Record that this node was accessed."""
        return self.evolve(
            accessed_at=datetime.utcnow(),
            access_count=self.access_count + 1,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "description": self.description,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "importance": self.importance.value if hasattr(self.importance, 'value') else self.importance,
            "state": self.state.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "memory_id": self.memory_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data["name"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
            description=data.get("description"),
            canonical_name=data.get("canonical_name"),
            aliases=set(data.get("aliases", [])),
            importance=MemoryImportance(data.get("importance", 5)),
            state=MemoryState[data.get("state", "ACTIVE")],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            accessed_at=datetime.fromisoformat(data["accessed_at"]) if "accessed_at" in data else datetime.utcnow(),
            access_count=data.get("access_count", 0),
            memory_id=data.get("memory_id"),
        )


@dataclass
class MemoryEdge:
    """
    An edge in the memory graph representing a relationship.
    
    Edges connect nodes and carry relationship semantics.
    Includes temporal validity intervals [valid_from, valid_until] for
    tracking when relationships are valid (e.g., CEO tenure periods).
    """
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    weight: float = 1.0
    confidence: float = 1.0
    importance: MemoryImportance = MemoryImportance.MEDIUM
    state: MemoryState = MemoryState.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    memory_id: Optional[str] = None
    # Temporal validity interval [valid_from, valid_until]
    # Used for tracking when facts are valid (e.g., "X was CEO from 2018 to 2023")
    valid_from: Optional[datetime] = None  # Start of validity (None = since creation)
    valid_until: Optional[datetime] = None  # End of validity (None = still valid)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from endpoints and relation."""
        content = f"{self.source_id}:{self.relation_type}:{self.target_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def evolve(self, **updates) -> "MemoryEdge":
        """Create a new edge with updates."""
        new_props = {**self.__dict__}
        new_props.update(updates)
        new_props["updated_at"] = datetime.utcnow()
        return MemoryEdge(**new_props)
    
    def strengthen(self, factor: float = 1.1) -> "MemoryEdge":
        """Strengthen the edge (increase weight and confidence)."""
        return self.evolve(
            weight=min(self.weight * factor, 10.0),
            confidence=min(self.confidence * factor, 1.0),
            accessed_at=datetime.utcnow(),
            access_count=self.access_count + 1,
        )
    
    def weaken(self, factor: float = 0.9) -> "MemoryEdge":
        """Weaken the edge (decay)."""
        return self.evolve(
            weight=max(self.weight * factor, 0.1),
        )
    
    def is_valid_at(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if this relationship is valid at a given time.
        
        Args:
            timestamp: Time to check (defaults to now)
        
        Returns:
            True if the relationship is valid at the given time
        
        Example:
            >>> # Check if someone was CEO in 2020
            >>> edge.is_valid_at(datetime(2020, 6, 1))
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Check start validity
        if self.valid_from and timestamp < self.valid_from:
            return False
        
        # Check end validity
        if self.valid_until and timestamp > self.valid_until:
            return False
        
        return True
    
    def supersede(self, end_time: Optional[datetime] = None) -> "MemoryEdge":
        """
        Mark this relationship as superseded (ended).
        
        Used when a fact changes (e.g., CEO transition).
        The relationship is preserved for historical queries.
        
        Args:
            end_time: When the relationship ended (defaults to now)
        
        Returns:
            Updated edge with valid_until set
        """
        return self.evolve(
            valid_until=end_time or datetime.utcnow(),
            state=MemoryState.ARCHIVED,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "description": self.description,
            "weight": self.weight,
            "confidence": self.confidence,
            "importance": self.importance.value if hasattr(self.importance, 'value') else self.importance,
            "state": self.state.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "memory_id": self.memory_id,
            # Temporal validity
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEdge":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            description=data.get("description"),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            importance=MemoryImportance(data.get("importance", 5)),
            state=MemoryState[data.get("state", "ACTIVE")],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            accessed_at=datetime.fromisoformat(data["accessed_at"]) if "accessed_at" in data else datetime.utcnow(),
            access_count=data.get("access_count", 0),
            memory_id=data.get("memory_id"),
            # Temporal validity
            valid_from=datetime.fromisoformat(data["valid_from"]) if data.get("valid_from") else None,
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
        )


@dataclass
class MemoryCluster:
    """
    A cluster of related memories forming a coherent topic.
    
    Clusters are discovered through community detection and
    provide high-level summaries of knowledge domains.
    """
    id: int
    summary: str
    entities: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    coherence_score: float = 1.0  # How well entities relate
    density: float = 1.0  # Edge density within cluster
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "summary": self.summary,
            "entities": self.entities,
            "edges": self.edges,
            "importance": self.importance.value if hasattr(self.importance, 'value') else self.importance,
            "coherence_score": self.coherence_score,
            "density": self.density,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "memory_id": self.memory_id,
            "metadata": self.metadata,
        }


@dataclass
class Memory:
    """
    A complete memory unit containing nodes, edges, and clusters.
    
    This is the primary unit of storage and retrieval in GraphMem.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Dict[str, MemoryNode] = field(default_factory=dict)
    edges: Dict[str, MemoryEdge] = field(default_factory=dict)
    clusters: Dict[int, MemoryCluster] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    state: MemoryState = MemoryState.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    parent_id: Optional[str] = None  # For tracking evolution
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def cluster_count(self) -> int:
        return len(self.clusters)
    
    def add_node(self, node: MemoryNode) -> None:
        """Add a node to this memory."""
        node.memory_id = self.id
        self.nodes[node.id] = node
        self.updated_at = datetime.utcnow()
    
    def add_edge(self, edge: MemoryEdge) -> None:
        """Add an edge to this memory."""
        edge.memory_id = self.id
        self.edges[edge.id] = edge
        self.updated_at = datetime.utcnow()
    
    def add_cluster(self, cluster: MemoryCluster) -> None:
        """Add a cluster to this memory."""
        cluster.memory_id = self.id
        self.clusters[cluster.id] = cluster
        self.updated_at = datetime.utcnow()
    
    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[MemoryEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_cluster(self, cluster_id: int) -> Optional[MemoryCluster]:
        """Get a cluster by ID."""
        return self.clusters.get(cluster_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "clusters": {k: v.to_dict() for k, v in self.clusters.items()},
            "metadata": self.metadata,
            "importance": self.importance.value if hasattr(self.importance, 'value') else self.importance,
            "state": self.state.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "parent_id": self.parent_id,
        }


@dataclass
class MemoryQuery:
    """
    A query against the memory system.
    
    Supports various query modes and filtering options.
    """
    query: str
    memory_id: Optional[str] = None
    mode: str = "semantic"  # semantic, exact, graph_traversal
    top_k: int = 10
    min_similarity: float = 0.5
    min_importance: MemoryImportance = MemoryImportance.EPHEMERAL
    include_clusters: bool = True
    include_context: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "memory_id": self.memory_id,
            "mode": self.mode,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "min_importance": self.min_importance.value if hasattr(self.min_importance, 'value') else self.min_importance,
            "include_clusters": self.include_clusters,
            "include_context": self.include_context,
            "filters": self.filters,
            "metadata": self.metadata,
        }


@dataclass
class MemoryResponse:
    """
    Response from a memory query.
    
    Contains retrieved context, confidence scores, and answer generation.
    """
    query: str
    answer: str
    confidence: float
    nodes: List[MemoryNode] = field(default_factory=list)
    edges: List[MemoryEdge] = field(default_factory=list)
    clusters: List[MemoryCluster] = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_results(self) -> bool:
        return bool(self.nodes or self.edges or self.clusters)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "clusters": [c.to_dict() for c in self.clusters],
            "context": self.context,
            "sources": self.sources,
            "metadata": self.metadata,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvolutionEvent:
    """
    A record of memory evolution (consolidation, decay, etc.).
    
    Provides audit trail for memory changes.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    evolution_type: EvolutionType = EvolutionType.CONSOLIDATION
    memory_id: str = ""
    affected_nodes: List[str] = field(default_factory=list)
    affected_edges: List[str] = field(default_factory=list)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evolution_type": self.evolution_type.name,
            "memory_id": self.memory_id,
            "affected_nodes": self.affected_nodes,
            "affected_edges": self.affected_edges,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "reason": self.reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }


# Type aliases for convenience
NodeId = str
EdgeId = str
ClusterId = int
MemoryId = str
Embedding = List[float]
Triplet = Tuple[str, str, str, Optional[str]]  # (source, relation, target, description)


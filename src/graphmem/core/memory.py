"""
GraphMem - Main Memory Class

The central interface for the GraphMem memory system.
Provides a simple, unified API for all memory operations.

Features:
- Automatic knowledge graph construction from documents
- Self-evolving memory with consolidation and decay
- Semantic and graph-based retrieval
- Multi-modal context engineering
- Production-ready with caching and persistence

Example:
    >>> from graphmem import GraphMem, MemoryConfig
    >>> 
    >>> # Initialize with defaults
    >>> memory = GraphMem()
    >>> 
    >>> # Or with custom configuration
    >>> config = MemoryConfig(
    ...     neo4j_uri="bolt://localhost:7687",
    ...     evolution_enabled=True,
    ...     consolidation_threshold=0.8,
    ... )
    >>> memory = GraphMem(config)
    >>> 
    >>> # Ingest documents
    >>> memory.ingest("Important document content...")
    >>> memory.ingest_file("report.pdf")
    >>> memory.ingest_url("https://example.com/article")
    >>> 
    >>> # Query the memory
    >>> response = memory.query("What are the key insights?")
    >>> print(response.answer)
    >>> 
    >>> # Evolve memory (consolidation, decay, synthesis)
    >>> memory.evolve()
"""

from __future__ import annotations
import os
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Callable, TypeVar
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from graphmem.core.memory_types import (
    Memory,
    MemoryNode,
    MemoryEdge,
    MemoryCluster,
    MemoryQuery,
    MemoryResponse,
    MemoryImportance,
    MemoryState,
    EvolutionEvent,
    EvolutionType,
)
from graphmem.core.exceptions import (
    GraphMemError,
    IngestionError,
    QueryError,
    StorageError,
    EvolutionError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MemoryConfig:
    """
    Configuration for GraphMem.
    
    All settings have sensible defaults for production use.
    Override as needed for your specific deployment.
    """
    
    # Storage backends
    neo4j_uri: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_PASSWORD", "password"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_DATABASE", "neo4j"))
    
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_REDIS_URL"))
    redis_ttl: int = 3600  # Cache TTL in seconds
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    llm_api_base: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_BASE"))  # For OpenRouter, Groq, etc.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 8000
    
    # Embedding Configuration
    embedding_provider: str = field(default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_PROVIDER", "openai"))
    embedding_model: str = field(default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"))
    embedding_api_base: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_API_BASE"))  # For OpenRouter, etc.
    embedding_dimensions: int = 1536
    
    # Extraction Configuration
    chunk_size: int = 2048
    chunk_overlap: int = 200
    max_triplets_per_chunk: int = 40
    extraction_workers: int = 8
    
    # Query Configuration
    similarity_top_k: int = 10
    min_similarity_threshold: float = 0.5
    max_context_length: int = 16000
    
    # Evolution Configuration
    evolution_enabled: bool = True
    consolidation_threshold: float = 0.85  # Similarity threshold for merging
    decay_enabled: bool = True
    decay_half_life_days: float = 30.0  # Time for memory strength to halve
    min_importance_to_keep: MemoryImportance = MemoryImportance.VERY_LOW
    rehydration_enabled: bool = True
    
    # Community Detection
    max_cluster_size: int = 100
    min_cluster_size: int = 2
    community_algorithm: str = "greedy_modularity"  # or "louvain", "label_propagation"
    
    # Retry and Resilience
    max_retries: int = 3
    retry_delay: float = 5.0
    connection_timeout: float = 30.0
    query_timeout: float = 60.0
    
    # Parallel Processing
    max_workers: int = 8
    batch_size: int = 500
    
    # Logging and Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False
    
    # Feature Flags
    enable_multimodal: bool = True  # Support images, audio, video
    enable_web_research: bool = True  # Internet research capability
    enable_synthetic_generation: bool = True  # Generate synthetic articles
    
    def validate(self) -> None:
        """Validate configuration and raise ConfigurationError if invalid."""
        if not self.neo4j_uri:
            raise ConfigurationError(
                "Neo4j URI is required",
                config_key="neo4j_uri",
            )
        
        if not self.llm_api_key:
            raise ConfigurationError(
                "LLM API key is required",
                config_key="llm_api_key",
                suggestions=["Set GRAPHMEM_LLM_API_KEY or OPENAI_API_KEY environment variable"],
            )
    
    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class GraphMem:
    """
    Production-Grade Agent Memory Framework.
    
    GraphMem provides a unified interface for building, querying, and evolving
    knowledge graphs that serve as long-term memory for AI agents.
    
    Key Features:
    - **Simple API**: Minimal learning curve, maximum power
    - **Self-Evolving**: Memory consolidates, decays, and improves automatically
    - **Production-Ready**: Built for scale with caching, persistence, and resilience
    - **Multi-Modal**: Supports text, PDFs, images, audio, video, and web pages
    - **Graph-Based**: Leverages knowledge graphs for rich, connected understanding
    
    Example:
        >>> memory = GraphMem()
        >>> 
        >>> # Ingest knowledge
        >>> memory.ingest("Tesla, led by CEO Elon Musk, is revolutionizing EVs...")
        >>> 
        >>> # Query with context
        >>> response = memory.query("Who leads Tesla?")
        >>> print(response.answer)  # "Elon Musk is the CEO of Tesla..."
        >>> 
        >>> # Memory evolves over time
        >>> memory.evolve()  # Consolidates related memories, decays old ones
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        memory_id: Optional[str] = None,
        auto_evolve: bool = False,
    ):
        """
        Initialize GraphMem.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
            memory_id: Optional ID for this memory instance. Auto-generated if not provided.
            auto_evolve: If True, memory evolves automatically on access.
        """
        self.config = config or MemoryConfig()
        self.memory_id = memory_id
        self.auto_evolve = auto_evolve
        
        self._initialized = False
        self._init_lock = threading.Lock()
        
        # Components (lazy-initialized)
        self._graph_store = None
        self._cache = None
        self._llm = None
        self._embeddings = None
        self._knowledge_graph = None
        self._entity_resolver = None
        self._community_detector = None
        self._retriever = None
        self._query_engine = None
        self._context_engine = None
        self._evolution_engine = None
        
        # Runtime state
        self._memory: Optional[Memory] = None
        self._evolution_history: List[EvolutionEvent] = []
        self._metrics: Dict[str, Any] = {
            "ingestions": 0,
            "queries": 0,
            "evolutions": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "total_clusters": 0,
        }
        
        # Background evolution thread
        self._evolution_thread: Optional[threading.Thread] = None
        self._evolution_stop_event = threading.Event()
        
        logger.info(f"GraphMem instance created (memory_id={self.memory_id})")
    
    @property
    def memory(self) -> Memory:
        """Access the underlying Memory object."""
        if self._memory is None:
            from uuid import uuid4
            self._memory = Memory(
                id=self.memory_id or str(uuid4()),
                name="GraphMem Memory",
                description="Auto-generated memory instance",
                created_at=datetime.utcnow(),
            )
            self.memory_id = self._memory.id
        return self._memory
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            try:
                self._initialize_components()
                self._initialized = True
                logger.info("GraphMem components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GraphMem: {e}")
                raise ConfigurationError(
                    f"Failed to initialize GraphMem: {e}",
                    cause=e,
                )
    
    def _initialize_components(self) -> None:
        """Initialize all components."""
        from graphmem.llm.providers import get_llm_provider
        from graphmem.llm.embeddings import get_embedding_provider
        from graphmem.stores.memory_store import InMemoryStore, InMemoryCache
        from graphmem.graph.knowledge_graph import KnowledgeGraph
        from graphmem.graph.entity_resolver import EntityResolver
        from graphmem.graph.community_detector import CommunityDetector
        from graphmem.retrieval.retriever import MemoryRetriever
        from graphmem.retrieval.query_engine import QueryEngine
        from graphmem.context.context_engine import ContextEngine
        from graphmem.evolution.memory_evolution import MemoryEvolution
        
        # Initialize LLM
        self._llm = get_llm_provider(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            api_base=self.config.llm_api_base,
        )
        
        # Initialize embeddings
        self._embeddings = get_embedding_provider(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
            api_key=self.config.embedding_api_key,
            api_base=self.config.embedding_api_base,
        )
        
        # Initialize storage (Neo4j if configured, otherwise in-memory)
        # Use Neo4j if any neo4j_uri is provided (including localhost)
        use_neo4j = bool(self.config.neo4j_uri)
        
        if use_neo4j:
            try:
                from graphmem.stores.neo4j_store import Neo4jStore
                self._graph_store = Neo4jStore(
                    uri=self.config.neo4j_uri,
                    username=self.config.neo4j_username,
                    password=self.config.neo4j_password,
                    database=self.config.neo4j_database,
                )
                logger.info("Using Neo4j for persistent storage")
            except Exception as e:
                logger.warning(f"Neo4j unavailable, falling back to in-memory: {e}")
                self._graph_store = InMemoryStore()
        else:
            self._graph_store = InMemoryStore()
            logger.info("Using in-memory storage (set neo4j_uri for persistence)")
        
        # Initialize cache (Redis if configured, otherwise in-memory)
        if self.config.redis_url:
            try:
                from graphmem.stores.redis_cache import RedisCache
                self._cache = RedisCache(
                    url=self.config.redis_url,
                    ttl=self.config.redis_ttl,
                )
                logger.info("Using Redis cache")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory cache: {e}")
                self._cache = InMemoryCache(ttl=self.config.redis_ttl)
        else:
            self._cache = InMemoryCache(ttl=self.config.redis_ttl)
        
        # Initialize entity resolver
        self._entity_resolver = EntityResolver(
            embeddings=self._embeddings,
            similarity_threshold=self.config.consolidation_threshold,
        )
        
        # Initialize knowledge graph
        self._knowledge_graph = KnowledgeGraph(
            llm=self._llm,
            embeddings=self._embeddings,
            store=self._graph_store,
            entity_resolver=self._entity_resolver,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            max_triplets_per_chunk=self.config.max_triplets_per_chunk,
        )
        
        # Initialize community detector
        self._community_detector = CommunityDetector(
            llm=self._llm,
            max_cluster_size=self.config.max_cluster_size,
            min_cluster_size=self.config.min_cluster_size,
            algorithm=self.config.community_algorithm,
        )
        
        # Initialize context engine
        self._context_engine = ContextEngine(
            llm=self._llm,
            embeddings=self._embeddings,
            token_limit=self.config.max_context_length,
        )
        
        # Initialize retriever
        self._retriever = MemoryRetriever(
            embeddings=self._embeddings,
            store=self._graph_store,
            cache=self._cache,
            top_k=self.config.similarity_top_k,
            min_similarity=self.config.min_similarity_threshold,
        )
        
        # Initialize query engine
        self._query_engine = QueryEngine(
            llm=self._llm,
            retriever=self._retriever,
            community_detector=self._community_detector,
            context_engine=self._context_engine,
        )
        
        # Initialize evolution engine
        if self.config.evolution_enabled:
            self._evolution_engine = MemoryEvolution(
                llm=self._llm,
                embeddings=self._embeddings,
                store=self._graph_store,
                consolidation_threshold=self.config.consolidation_threshold,
                decay_enabled=self.config.decay_enabled,
                decay_half_life_days=self.config.decay_half_life_days,
            )
        
        # Initialize or load memory
        if self.memory_id:
            self._load_memory()
        else:
            self._create_new_memory()
    
    def _create_new_memory(self) -> None:
        """Create a new memory instance."""
        from uuid import uuid4
        self._memory = Memory(
            id=str(uuid4()),
            name="GraphMem Memory",
            description="Auto-generated memory instance",
            created_at=datetime.utcnow(),
        )
        self.memory_id = self._memory.id
        logger.info(f"Created new memory: {self.memory_id}")
    
    def _load_memory(self) -> None:
        """Load existing memory from storage."""
        try:
            loaded = self._graph_store.load_memory(self.memory_id)
            if loaded:
                self._memory = loaded
                logger.info(f"Loaded memory: {self.memory_id}")
            else:
                self._create_new_memory()
        except Exception as e:
            logger.warning(f"Failed to load memory {self.memory_id}: {e}")
            self._create_new_memory()
    
    # ==================== PUBLIC API ====================
    
    def ingest(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest text content into memory.
        
        Extracts entities, relationships, and builds knowledge graph.
        
        Args:
            content: Text content to ingest
            metadata: Optional metadata to attach
            importance: Importance level for decay prioritization
            progress_callback: Optional callback for progress updates (stage, percent)
        
        Returns:
            Dict with ingestion statistics
        
        Example:
            >>> result = memory.ingest(
            ...     "Apple Inc. was founded by Steve Jobs in 1976...",
            ...     metadata={"source": "wikipedia"},
            ...     importance=MemoryImportance.HIGH,
            ... )
            >>> print(f"Extracted {result['entities']} entities")
        """
        self._ensure_initialized()
        
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback("parsing", 0.1)
            
            # Build knowledge graph from content
            nodes, edges = self._knowledge_graph.extract(
                content=content,
                metadata=metadata or {},
                memory_id=self.memory_id,
                progress_callback=progress_callback,
            )
            
            if progress_callback:
                progress_callback("resolving_entities", 0.6)
            
            # Add to memory
            for node in nodes:
                node.importance = importance
                self._memory.add_node(node)
            
            for edge in edges:
                edge.importance = importance
                self._memory.add_edge(edge)
            
            if progress_callback:
                progress_callback("building_communities", 0.8)
            
            # Rebuild communities
            clusters = self._community_detector.detect(
                nodes=list(self._memory.nodes.values()),
                edges=list(self._memory.edges.values()),
                memory_id=self.memory_id,
            )
            
            for cluster in clusters:
                self._memory.add_cluster(cluster)
            
            if progress_callback:
                progress_callback("persisting", 0.9)
            
            # Persist to storage
            self._graph_store.save_memory(self._memory)
            
            # Invalidate cache
            if self._cache:
                self._cache.invalidate(self.memory_id)
            
            elapsed = time.time() - start_time
            
            # Update metrics
            self._metrics["ingestions"] += 1
            self._metrics["total_nodes"] = len(self._memory.nodes)
            self._metrics["total_edges"] = len(self._memory.edges)
            self._metrics["total_clusters"] = len(self._memory.clusters)
            
            if progress_callback:
                progress_callback("complete", 1.0)
            
            result = {
                "success": True,
                "memory_id": self.memory_id,
                "entities": len(nodes),
                "relationships": len(edges),
                "clusters": len(clusters),
                "elapsed_seconds": elapsed,
            }
            
            logger.info(f"Ingested content: {len(nodes)} entities, {len(edges)} relationships")
            
            # Auto-evolve if enabled
            if self.auto_evolve:
                self.evolve()
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise IngestionError(
                f"Failed to ingest content: {e}",
                stage="extraction",
                cause=e,
            )
    
    def ingest_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a file into memory.
        
        Supports PDF, images, audio, video, and text files.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata
            importance: Importance level
            progress_callback: Progress callback
        
        Returns:
            Ingestion statistics
        """
        self._ensure_initialized()
        
        try:
            # Extract content using context engine
            content = self._context_engine.extract_from_file(file_path)
            
            # Add file metadata
            file_metadata = metadata or {}
            file_metadata["source_file"] = str(file_path)
            file_metadata["source_type"] = "file"
            
            return self.ingest(
                content=content,
                metadata=file_metadata,
                importance=importance,
                progress_callback=progress_callback,
            )
            
        except Exception as e:
            logger.error(f"File ingestion failed: {e}")
            raise IngestionError(
                f"Failed to ingest file {file_path}: {e}",
                cause=e,
            )
    
    def ingest_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest content from a URL.
        
        Supports web pages and YouTube videos.
        
        Args:
            url: URL to fetch and ingest
            metadata: Optional metadata
            importance: Importance level
            progress_callback: Progress callback
        
        Returns:
            Ingestion statistics
        """
        self._ensure_initialized()
        
        try:
            # Extract content using context engine
            content = self._context_engine.extract_from_url(url)
            
            # Add URL metadata
            url_metadata = metadata or {}
            url_metadata["source_url"] = url
            url_metadata["source_type"] = "url"
            
            return self.ingest(
                content=content,
                metadata=url_metadata,
                importance=importance,
                progress_callback=progress_callback,
            )
            
        except Exception as e:
            logger.error(f"URL ingestion failed: {e}")
            raise IngestionError(
                f"Failed to ingest URL {url}: {e}",
                cause=e,
            )
    
    def query(
        self,
        query: str,
        mode: str = "semantic",
        top_k: int = 10,
        include_context: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Query the memory.
        
        Args:
            query: Natural language query
            mode: Query mode - "semantic", "exact", or "graph_traversal"
            top_k: Maximum results to consider
            include_context: Whether to include surrounding context
            filters: Optional filters for results
        
        Returns:
            MemoryResponse with answer, confidence, and supporting context
        
        Example:
            >>> response = memory.query("Who founded Apple?")
            >>> print(response.answer)
            >>> print(f"Confidence: {response.confidence}")
            >>> for node in response.nodes:
            ...     print(f"- {node.name}: {node.description}")
        """
        self._ensure_initialized()
        
        try:
            start_time = time.time()
            
            # Build query object
            memory_query = MemoryQuery(
                query=query,
                memory_id=self.memory_id,
                mode=mode,
                top_k=top_k,
                include_context=include_context,
                filters=filters or {},
            )
            
            # Execute query
            response = self._query_engine.query(
                query=memory_query,
                memory=self._memory,
            )
            
            # Update metrics
            response.latency_ms = (time.time() - start_time) * 1000
            self._metrics["queries"] += 1
            
            # Record access on retrieved nodes
            for node in response.nodes:
                if node.id in self._memory.nodes:
                    self._memory.nodes[node.id] = node.record_access()
            
            logger.info(f"Query completed: '{query[:50]}...' -> {len(response.nodes)} nodes")
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(
                f"Query failed: {e}",
                query=query,
                cause=e,
            )
    
    def evolve(
        self,
        evolution_types: Optional[List[EvolutionType]] = None,
        force: bool = False,
    ) -> List[EvolutionEvent]:
        """
        Evolve the memory.
        
        Performs consolidation, decay, and synthesis to improve memory quality.
        This is how the memory becomes "self-improving" like human memory.
        
        Args:
            evolution_types: Specific evolution types to run. If None, runs all enabled.
            force: If True, runs even if recently evolved.
        
        Returns:
            List of evolution events that occurred
        
        Example:
            >>> # Evolve all aspects
            >>> events = memory.evolve()
            >>> 
            >>> # Only consolidate
            >>> events = memory.evolve([EvolutionType.CONSOLIDATION])
            >>> 
            >>> for event in events:
            ...     print(f"{event.evolution_type}: {len(event.affected_nodes)} nodes")
        """
        self._ensure_initialized()
        
        if not self.config.evolution_enabled:
            logger.info("Evolution is disabled")
            return []
        
        if not self._evolution_engine:
            logger.warning("Evolution engine not initialized")
            return []
        
        try:
            events = self._evolution_engine.evolve(
                memory=self._memory,
                evolution_types=evolution_types,
                force=force,
            )
            
            # Update memory with evolution results
            if events:
                self._graph_store.save_memory(self._memory)
                
                if self._cache:
                    self._cache.invalidate(self.memory_id)
            
            self._evolution_history.extend(events)
            self._metrics["evolutions"] += len(events)
            
            logger.info(f"Evolution completed: {len(events)} events")
            
            return events
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise EvolutionError(
                f"Evolution failed: {e}",
                cause=e,
            )
    
    def rehydrate(
        self,
        context: str,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Rehydrate memory with context.
        
        Strengthens memories relevant to the given context and
        potentially restores archived memories.
        
        Args:
            context: Context to use for rehydration
            max_nodes: Maximum nodes to rehydrate
        
        Returns:
            Rehydration statistics
        """
        self._ensure_initialized()
        
        if not self._evolution_engine:
            return {"rehydrated": 0, "restored": 0}
        
        try:
            return self._evolution_engine.rehydrate(
                memory=self._memory,
                context=context,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Rehydration failed: {e}")
            raise EvolutionError(
                f"Rehydration failed: {e}",
                evolution_type="rehydration",
                cause=e,
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        self._ensure_initialized()
        
        return {
            "memory_id": self.memory_id,
            "nodes": len(self._memory.nodes),
            "edges": len(self._memory.edges),
            "clusters": len(self._memory.clusters),
            "created_at": self._memory.created_at.isoformat(),
            "updated_at": self._memory.updated_at.isoformat(),
            "version": self._memory.version,
            "metrics": self._metrics.copy(),
            "evolution_history": len(self._evolution_history),
        }
    
    def get_graph(self) -> Dict[str, Any]:
        """
        Get the full knowledge graph.
        
        Returns entities, relationships, and clusters.
        """
        self._ensure_initialized()
        
        return {
            "memory_id": self.memory_id,
            "nodes": [n.to_dict() for n in self._memory.nodes.values()],
            "edges": [e.to_dict() for e in self._memory.edges.values()],
            "clusters": [c.to_dict() for c in self._memory.clusters.values()],
        }
    
    def clear(self) -> None:
        """Clear all memory data."""
        self._ensure_initialized()
        
        try:
            self._graph_store.clear_memory(self.memory_id)
            
            if self._cache:
                self._cache.invalidate(self.memory_id)
            
            self._create_new_memory()
            self._evolution_history.clear()
            
            logger.info(f"Memory cleared: {self.memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            raise StorageError(
                f"Failed to clear memory: {e}",
                storage_type="neo4j",
                operation="clear",
                cause=e,
            )
    
    def save(self) -> None:
        """Save memory to persistent storage."""
        self._ensure_initialized()
        
        try:
            self._graph_store.save_memory(self._memory)
            logger.info(f"Memory saved: {self.memory_id}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            raise StorageError(
                f"Failed to save memory: {e}",
                storage_type="neo4j",
                operation="save",
                cause=e,
            )
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
        try:
            if self._evolution_stop_event:
                self._evolution_stop_event.set()
            
            if self._graph_store:
                self._graph_store.close()
            
            if self._cache:
                self._cache.close()
            
            logger.info("GraphMem closed")
        except Exception as e:
            logger.error(f"Error closing GraphMem: {e}")
    
    def __enter__(self) -> "GraphMem":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        if self._memory:
            return f"GraphMem(id={self.memory_id}, nodes={len(self._memory.nodes)}, edges={len(self._memory.edges)})"
        return f"GraphMem(id={self.memory_id}, initialized={self._initialized})"


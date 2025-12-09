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
    >>> # Or with custom configuration (Neo4j is optional)
    >>> config = MemoryConfig(
    ...     neo4j_uri="neo4j+s://your-instance.neo4j.io",  # Optional - uses in-memory if not set
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
    
    Multi-Tenant Notes:
        - Set `user_id` to isolate data per user/tenant
        - Each user can have multiple memory sessions via `memory_id`
        - Data is filtered by BOTH user_id AND memory_id
    """
    
    # Multi-tenant isolation (IMPORTANT for production!)
    user_id: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_USER_ID"))
    
    # Storage backends (None = use in-memory storage)
    neo4j_uri: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_URI", None))
    neo4j_username: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_USERNAME", "neo4j"))
    neo4j_password: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_PASSWORD", None))
    neo4j_database: str = field(default_factory=lambda: os.getenv("GRAPHMEM_NEO4J_DATABASE", "neo4j"))
    
    redis_url: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_REDIS_URL"))
    redis_ttl: int = 3600  # Cache TTL in seconds
    
    # Turso Configuration (SQLite-based alternative to Neo4j + Redis)
    turso_db_path: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_TURSO_DB_PATH"))
    turso_url: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_TURSO_URL"))  # For cloud sync
    turso_auth_token: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_TURSO_AUTH_TOKEN"))
    
    # LLM Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_MODEL", "gpt-4o-mini"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    llm_api_base: Optional[str] = field(default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_BASE"))  # For OpenRouter, Groq, etc.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 8000
    
    # Azure OpenAI Configuration (used when llm_provider="azure_openai")
    azure_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    azure_deployment: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT"))  # LLM deployment name
    azure_embedding_deployment: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_EMBEDDING_DEPLOYMENT"))  # Embedding deployment
    
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
        # neo4j_uri is optional - if not provided, in-memory storage is used
        
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
        user_id: Optional[str] = None,
        auto_evolve: bool = False,
    ):
        """
        Initialize GraphMem.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
            memory_id: ID for this memory session. **REQUIRED for persistence!**
                       If None, a new UUID is generated each time (data won't persist across sessions).
            user_id: Optional user/tenant ID for multi-tenant isolation.
                     If not provided, uses config.user_id or falls back to "default".
            auto_evolve: If True, memory evolves automatically on access.
        
        !!! warning "Important: Provide memory_id for Persistence"
            Without a consistent `memory_id`, you'll get a new UUID each time and won't find your old data!
            
            ```python
            # ✅ CORRECT: Data persists
            memory = GraphMem(config, memory_id="my_agent", user_id="user1")
            
            # ❌ WRONG: New UUID each time, data lost!
            memory = GraphMem(config)  # No memory_id
            ```
        
        Multi-Tenant Usage:
            # User A's memory
            gm_alice = GraphMem(config, user_id="alice", memory_id="chat_1")
            
            # User B's memory (isolated from Alice)
            gm_bob = GraphMem(config, user_id="bob", memory_id="chat_1")  # Same memory_id, different user
        """
        self.config = config or MemoryConfig()
        self.memory_id = memory_id
        
        # Multi-tenant isolation: user_id separates different users' data
        self.user_id = user_id or self.config.user_id or "default"
        
        self.auto_evolve = auto_evolve
        
        self._initialized = False
        self._init_lock = threading.Lock()
        self._memory_lock = threading.Lock()  # Thread-safe access to memory dictionaries
        
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
        
        logger.info(f"GraphMem instance created (memory_id={self.memory_id}, user_id={self.user_id})")
        
        # Warn if memory_id is None (data won't persist)
        if self.memory_id is None:
            logger.warning(
                "memory_id is None - a new UUID will be generated. "
                "Your data won't persist across sessions! "
                "Provide memory_id='your_id' to enable persistence."
            )
    
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
        llm_kwargs = {
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "api_key": self.config.llm_api_key,
            "api_base": self.config.llm_api_base,
        }
        # Add Azure-specific config if using Azure
        if self.config.llm_provider == "azure_openai":
            llm_kwargs["api_version"] = self.config.azure_api_version
            llm_kwargs["deployment"] = self.config.azure_deployment or self.config.llm_model
        
        self._llm = get_llm_provider(**llm_kwargs)
        
        # Initialize cache (Redis > Turso > InMemory)
        # Cache must be initialized BEFORE embeddings so embeddings can use it
        if self.config.redis_url:
            try:
                from graphmem.stores.redis_cache import RedisCache
                self._cache = RedisCache(
                    url=self.config.redis_url,
                    ttl=self.config.redis_ttl,
                )
                logger.info("Using Redis cache")
            except Exception as e:
                logger.warning(f"Redis unavailable, trying Turso cache: {e}")
                self._cache = None
        else:
            self._cache = None
        
        # Fall back to Turso cache if Redis not configured/available
        if self._cache is None and self.config.turso_db_path:
            try:
                from graphmem.stores.turso_store import TursoCache, TURSO_AVAILABLE
                if TURSO_AVAILABLE:
                    cache_path = f"{self.config.turso_db_path}_cache.db"
                    self._cache = TursoCache(
                        db_path=cache_path,
                        ttl=self.config.redis_ttl,
                        turso_url=self.config.turso_url,
                        turso_auth_token=self.config.turso_auth_token,
                    )
                    logger.info(f"Using Turso cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Turso cache unavailable: {e}")
        
        # Final fallback to in-memory cache
        if self._cache is None:
            self._cache = InMemoryCache(ttl=self.config.redis_ttl)
        
        # Initialize embeddings (with cache for embedding reuse)
        emb_kwargs = {
            "provider": self.config.embedding_provider,
            "model": self.config.embedding_model,
            "api_key": self.config.embedding_api_key,
            "api_base": self.config.embedding_api_base,
            "cache": self._cache,  # Pass cache for embedding caching
        }
        # Add Azure-specific config if using Azure embeddings
        if self.config.embedding_provider == "azure_openai":
            emb_kwargs["api_version"] = self.config.azure_api_version
            emb_kwargs["deployment"] = self.config.azure_embedding_deployment or self.config.embedding_model
        
        self._embeddings = get_embedding_provider(**emb_kwargs)
        
        # Initialize storage (Neo4j > Turso > InMemory)
        # Priority: Neo4j (full graph), Turso (SQLite persistent), InMemory (default)
        use_neo4j = bool(self.config.neo4j_uri)
        use_turso = bool(self.config.turso_db_path) and not use_neo4j
        
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
                logger.warning(f"Neo4j unavailable, trying Turso: {e}")
                use_turso = bool(self.config.turso_db_path)
                if not use_turso:
                    self._graph_store = InMemoryStore()
        
        if use_turso and self._graph_store is None:
            try:
                from graphmem.stores.turso_store import TursoStore, TURSO_AVAILABLE
                if TURSO_AVAILABLE:
                    self._graph_store = TursoStore(
                        db_path=self.config.turso_db_path,
                        turso_url=self.config.turso_url,
                        turso_auth_token=self.config.turso_auth_token,
                    )
                    logger.info(f"Using Turso for persistent storage: {self.config.turso_db_path}")
                else:
                    logger.warning("Turso not available (pip install libsql-experimental)")
                    self._graph_store = InMemoryStore()
            except Exception as e:
                logger.warning(f"Turso unavailable, falling back to in-memory: {e}")
                self._graph_store = InMemoryStore()
        
        if self._graph_store is None:
            self._graph_store = InMemoryStore()
            logger.info("Using in-memory storage (set neo4j_uri or turso_db_path for persistence)")
        
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
        
        # Initialize retriever (with Neo4j vector search if available)
        self._retriever = MemoryRetriever(
            embeddings=self._embeddings,
            store=self._graph_store,
            cache=self._cache,
            llm=self._llm,  # For LLM-based alias expansion during retrieval
            top_k=self.config.similarity_top_k,
            min_similarity=self.config.min_similarity_threshold,
            memory_id=self.memory_id,  # For Neo4j vector search
            user_id=self.user_id,  # Multi-tenant isolation
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
        # Use provided memory_id or generate a new one
        memory_id = self.memory_id or str(uuid4())
        self._memory = Memory(
            id=memory_id,
            name="GraphMem Memory",
            description="Auto-generated memory instance",
            created_at=datetime.utcnow(),
        )
        self.memory_id = self._memory.id
        logger.info(f"Created new memory: {self.memory_id}")
    
    def _load_memory(self) -> None:
        """Load existing memory from storage (filtered by user_id for multi-tenant isolation)."""
        try:
            loaded = self._graph_store.load_memory(self.memory_id, self.user_id)
            if loaded:
                self._memory = loaded
                logger.info(f"Loaded memory: {self.memory_id} for user: {self.user_id}")
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
            
            # Get existing nodes for coreference resolution (cross-document entity linking)
            existing_nodes = list(self._memory.nodes.values()) if self._memory.nodes else []
            
            # Build knowledge graph from content with coreference resolution
            nodes, edges = self._knowledge_graph.extract(
                content=content,
                metadata=metadata or {},
                memory_id=self.memory_id,
                user_id=self.user_id,  # Multi-tenant isolation
                progress_callback=progress_callback,
                existing_nodes=existing_nodes,  # NEW: For coreference resolution
                enable_coreference=len(existing_nodes) > 0,  # Only if we have existing entities
            )
            
            if progress_callback:
                progress_callback("resolving_entities", 0.6)
            
            # Thread-safe memory modification
            with self._memory_lock:
                # FIRST: Snapshot existing data before any modifications
                # This avoids "dictionary changed size during iteration" errors
                existing_node_ids = set(self._memory.nodes.keys())
                existing_edge_ids = set(self._memory.edges.keys())
            
            # Add to memory
            for node in nodes:
                node.importance = importance
                self._memory.add_node(node)
            
            for edge in edges:
                edge.importance = importance
                self._memory.add_edge(edge)
            
            if progress_callback:
                progress_callback("building_communities", 0.8)
            
            # Create safe copies for community detection
            # Use dict.copy() + values() which is safer than list(values())
            nodes_snapshot = list(self._memory.nodes.copy().values())
            edges_snapshot = list(self._memory.edges.copy().values())
            
            # Rebuild communities with snapshot copies
            clusters = self._community_detector.detect(
                nodes=nodes_snapshot,
                edges=edges_snapshot,
                memory_id=self.memory_id,
            )
            
            for cluster in clusters:
                self._memory.add_cluster(cluster)
            
            if progress_callback:
                progress_callback("persisting", 0.9)
            
            # Persist to storage
            self._graph_store.save_memory(self._memory)
            
            # Invalidate cache (multi-tenant safe)
            if self._cache:
                self._cache.invalidate(self.memory_id, user_id=self.user_id)
            
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
    
    def ingest_batch(
        self,
        documents: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        auto_scale: bool = True,
        aggressive: bool = False,
        rebuild_communities: bool = True,
    ) -> Dict[str, Any]:
        """
        High-performance batch ingestion using concurrent processing.
        
        Uses ThreadPoolExecutor for parallel document ingestion.
        Auto-detects optimal worker count based on hardware and API limits.
        
        Args:
            documents: List of documents with {"id": str, "content": str, ...}
            max_workers: Number of concurrent workers (None = auto-detect)
            show_progress: Show progress logs
            auto_scale: Auto-detect optimal workers based on hardware/provider
        
        Returns:
            Batch ingestion statistics
        
        Example:
            >>> docs = [
            ...     {"id": "doc1", "content": "Apple was founded..."},
            ...     {"id": "doc2", "content": "Microsoft was founded..."},
            ... ]
            >>> # Auto-detect optimal workers
            >>> result = memory.ingest_batch(docs)
            >>> # Or specify manually
            >>> result = memory.ingest_batch(docs, max_workers=20)
        """
        self._ensure_initialized()
        
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        start_time = time.time()
        
        # Auto-detect optimal workers if not specified
        if max_workers is None and auto_scale:
            try:
                from graphmem.ingestion import AutoScaler
                scaler = AutoScaler()
                config = scaler.get_optimal_config(
                    provider=self.config.llm_provider,
                    aggressive=aggressive,
                )
                max_workers = config.extraction_workers
                mode = "AGGRESSIVE" if aggressive else "normal"
                if show_progress:
                    logger.info(f"🔧 Auto-detected workers: {max_workers} ({mode} mode, provider: {self.config.llm_provider})")
            except Exception as e:
                logger.warning(f"Auto-scale failed, using default: {e}")
                max_workers = 10 if aggressive else 5
        elif max_workers is None:
            max_workers = 10 if aggressive else 5
        
        # Validate and filter documents - skip invalid entries (None, ellipsis, non-dict)
        valid_documents = []
        skipped = 0
        for i, doc in enumerate(documents):
            if doc is None or doc is ... or not isinstance(doc, dict):
                logger.warning(f"   ⚠️ Skipping invalid document at index {i}: {type(doc).__name__}")
                skipped += 1
                continue
            # Also check for empty content
            content = doc.get("content", doc.get("text", ""))
            if not content or not isinstance(content, str) or not content.strip():
                logger.warning(f"   ⚠️ Skipping document at index {i}: empty or invalid content")
                skipped += 1
                continue
            valid_documents.append(doc)
        
        if skipped > 0:
            logger.info(f"   📋 Filtered {skipped} invalid documents, processing {len(valid_documents)}")
        
        if not valid_documents:
            logger.warning("   ❌ No valid documents to ingest!")
            return {
                "success": False,
                "documents_processed": 0,
                "documents_failed": 0,
                "documents_skipped": skipped,
                "total_entities": 0,
                "total_relationships": 0,
                "clusters_built": 0,
                "elapsed_seconds": 0,
                "throughput_docs_per_sec": 0,
            }
        
        if show_progress:
            logger.info(f"🚀 Batch ingesting {len(valid_documents)} documents with {max_workers} workers")
        
        # Thread-safe counters
        results_lock = threading.Lock()
        processed = 0
        failed = 0
        total_entities = 0
        total_relationships = 0
        
        def ingest_single(doc):
            """
            Fast-path ingestion for a single document:
            - Extract (LLM + embeddings) off-lock
            - Accumulate nodes/edges for batch merge
            - Infinite retry on rate limits
            """
            nonlocal processed, failed, total_entities, total_relationships
            
            # Defensive check (already validated, but just in case)
            if not isinstance(doc, dict):
                with results_lock:
                    failed += 1
                return {"success": False, "doc_id": "invalid", "error": f"Invalid document type: {type(doc).__name__}", "nodes": [], "edges": []}
            
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", doc.get("text", ""))
            metadata = {k: v for k, v in doc.items() if k not in ("id", "content", "text")}
            
            base_delay = 2  # seconds
            max_delay = 60  # cap at 60 seconds
            attempt = 0
            
            while True:
                try:
                    # Get snapshot of existing nodes for coreference (thread-safe read)
                    existing_nodes = list(self._memory.nodes.values()) if self._memory.nodes else []
                    
                    nodes, edges = self._knowledge_graph.extract(
                        content=content,
                        metadata=metadata,
                        memory_id=self.memory_id,
                        user_id=self.user_id,
                        progress_callback=None,
                        existing_nodes=existing_nodes,  # For coreference resolution
                        enable_coreference=len(existing_nodes) > 0,
                    )
                    
                    with results_lock:
                        processed += 1
                        total_entities += len(nodes)
                        total_relationships += len(edges)
                    
                    return {"success": True, "doc_id": doc_id, "nodes": nodes, "edges": edges}
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit error (429)
                    is_rate_limit = any(x in error_str for x in ['429', 'rate limit', 'too many requests', 'quota', 'retry'])
                    
                    if is_rate_limit:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        attempt += 1
                        if show_progress:
                            logger.warning(f"   ⏳ Rate limit hit for {doc_id}, retry #{attempt} in {delay}s...")
                        time.sleep(delay)
                        continue  # Retry forever
                    
                    # Not a rate limit error - actual failure
                    with results_lock:
                        failed += 1
                    
                    logger.warning(f"   ⚠️ Failed to ingest {doc_id}: {str(e)[:160]}")
                    return {"success": False, "doc_id": doc_id, "error": str(e), "nodes": [], "edges": []}
        
        # Process documents in parallel (using validated documents)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(ingest_single, doc): i for i, doc in enumerate(valid_documents)}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if show_progress and len(results) % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = len(results) / elapsed if elapsed > 0 else 0
                        logger.info(f"   Progress: {len(results)}/{len(valid_documents)} ({rate:.1f} docs/sec)")
                        
                except Exception as e:
                    logger.error(f"   Document {idx} failed: {e}")
                    failed += 1
        
        # Merge all successful nodes/edges with a single lock and optional community rebuild
        merged_nodes = []
        merged_edges = []
        for r in results:
            if r.get("success"):
                merged_nodes.extend(r.get("nodes", []))
                merged_edges.extend(r.get("edges", []))
        
        clusters = []
        if merged_nodes or merged_edges:
            with self._memory_lock:
                # Add nodes/edges
                for node in merged_nodes:
                    self._memory.add_node(node)
                for edge in merged_edges:
                    self._memory.add_edge(edge)
                
                # Optionally rebuild communities once per batch
                if rebuild_communities:
                    clusters = self._community_detector.detect(
                        nodes=list(self._memory.nodes.copy().values()),
                        edges=list(self._memory.edges.copy().values()),
                        memory_id=self.memory_id,
                    )
                    for cluster in clusters:
                        self._memory.add_cluster(cluster)
                
                # Persist to storage
                self._graph_store.save_memory(self._memory)
                
                # Invalidate cache (multi-tenant safe)
                if self._cache:
                    self._cache.invalidate(self.memory_id, user_id=self.user_id)
                
                # Update metrics
                self._metrics["ingestions"] += processed
                self._metrics["total_nodes"] = len(self._memory.nodes)
                self._metrics["total_edges"] = len(self._memory.edges)
                self._metrics["total_clusters"] = len(self._memory.clusters)
        
        elapsed = time.time() - start_time
        throughput = len(valid_documents) / elapsed if elapsed > 0 else 0
        
        if show_progress:
            skip_info = f", {skipped} skipped" if skipped > 0 else ""
            logger.info(f"✅ Batch complete: {processed} processed, {failed} failed{skip_info} in {elapsed:.1f}s ({throughput:.1f} docs/sec)")
        
        # Auto-evolve if enabled
        if self.auto_evolve and processed > 0:
            self.evolve()
        
        return {
            "success": failed == 0 and processed > 0,
            "documents_processed": processed,
            "documents_failed": failed,
            "documents_skipped": skipped,
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "clusters_built": len(clusters),
            "elapsed_seconds": elapsed,
            "throughput_docs_per_sec": throughput,
        }
    
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
            
            # Thread-safe query execution (reads memory dictionaries)
            with self._memory_lock:
                # Execute query
                response = self._query_engine.query(
                    query=memory_query,
                    memory=self._memory,
                )
                
                # Record access on retrieved nodes
                for node in response.nodes:
                    if node.id in self._memory.nodes:
                        self._memory.nodes[node.id] = node.record_access()
                
                # Update metrics
                response.latency_ms = (time.time() - start_time) * 1000
                self._metrics["queries"] += 1
                
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
            # Thread-safe evolution (modifies memory dictionaries)
            with self._memory_lock:
                events = self._evolution_engine.evolve(
                    memory=self._memory,
                    evolution_types=evolution_types,
                    force=force,
                )
                
                # Update memory with evolution results
                if events:
                    self._graph_store.save_memory(self._memory)
                    
                    if self._cache:
                        self._cache.invalidate(self.memory_id, user_id=self.user_id)
                
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
        
        # Thread-safe iteration - use dict.copy() to avoid concurrent modification
        with self._memory_lock:
            nodes_copy = self._memory.nodes.copy()
            edges_copy = self._memory.edges.copy()
            clusters_copy = self._memory.clusters.copy()
        
        return {
            "memory_id": self.memory_id,
            "nodes": [n.to_dict() for n in nodes_copy.values()],
            "edges": [e.to_dict() for e in edges_copy.values()],
            "clusters": [c.to_dict() for c in clusters_copy.values()],
        }
    
    def clear(self) -> None:
        """Clear all memory data."""
        self._ensure_initialized()
        
        try:
            self._graph_store.clear_memory(self.memory_id)
            
            if self._cache:
                self._cache.invalidate(self.memory_id, user_id=self.user_id)
            
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


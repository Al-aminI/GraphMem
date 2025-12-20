"""
GraphMem Turso Store

SQLite-based storage backend using Turso/libSQL.
Provides persistence, vector search, and optional cloud sync.

Benefits over InMemoryStore:
- Data persists across restarts (SQLite file)
- Native vector similarity search
- Offline-first with optional cloud sync
- Per-user database files for true isolation
"""

from __future__ import annotations
import logging
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable, TypeVar

from graphmem.core.memory_types import Memory, MemoryNode, MemoryEdge, MemoryCluster, MemoryImportance

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _is_stream_error(error: Exception) -> bool:
    """Check if the error is a stale stream/connection error from Turso."""
    error_str = str(error).lower()
    return any(msg in error_str for msg in [
        "stream not found",
        "connection closed",
        "connection reset",
        "websocket",
        "hrana",
        "peer closed connection",
    ])

# Try to import libsql
try:
    import libsql
    TURSO_AVAILABLE = True
except ImportError:
    TURSO_AVAILABLE = False
    logger.warning("libsql not installed. Install with: pip install libsql")


class TursoStore:
    """
    Turso/libSQL storage backend for GraphMem.
    
    Perfect for:
    - Persistent local storage (survives restarts)
    - Offline-first AI agents
    - Edge deployments
    - Cost-effective alternative to Neo4j
    - Per-user database isolation
    
    Features:
    - SQLite-compatible (libSQL)
    - Native vector similarity search
    - Optional sync to Turso Cloud
    - Multi-tenant support
    """
    
    def __init__(
        self,
        db_path: str = "graphmem.db",
        turso_url: Optional[str] = None,
        turso_auth_token: Optional[str] = None,
        sync_mode: str = "full",  # "full", "push", "pull", or None
        embedding_dimensions: int = 1536,  # OpenAI text-embedding-3-small default
    ):
        """
        Initialize Turso store.
        
        Args:
            db_path: Local SQLite file path
            turso_url: Optional Turso Cloud URL for sync
            turso_auth_token: Auth token for Turso Cloud
            sync_mode: Sync mode - "full" (bidirectional), "push", "pull", or None
        """
        if not TURSO_AVAILABLE:
            raise ImportError(
                "libsql_experimental is required for TursoStore. "
                "Install with: pip install libsql-experimental"
            )
        
        self.db_path = db_path
        self.turso_url = turso_url
        self.turso_auth_token = turso_auth_token
        self.sync_mode = sync_mode
        self.embedding_dimensions = embedding_dimensions
        
        # Connection retry settings (infinite with exponential backoff)
        self._base_retry_delay = 1.0  # Initial delay in seconds
        self._max_retry_delay = 30.0  # Cap backoff at 30 seconds
        
        # Connect to database
        self._connect()
        
        # Initialize schema
        self._init_schema()
    
    def _connect(self) -> None:
        """Establish connection to Turso database."""
        if self.turso_url and self.turso_auth_token:
            # Cloud-synced mode
            self.conn = libsql.connect(
                self.db_path,
                sync_url=self.turso_url,
                auth_token=self.turso_auth_token,
            )
            logger.info(f"TursoStore connected with cloud sync: {self.turso_url}")
        else:
            # Local-only mode
            self.conn = libsql.connect(self.db_path)
            logger.info(f"TursoStore connected (local): {self.db_path}")
    
    def _reconnect(self) -> None:
        """Reconnect to database after a stale connection error."""
        logger.warning("TursoStore: Reconnecting due to stale connection...")
        try:
            # Try to close old connection gracefully
            if hasattr(self, 'conn') and self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass  # Ignore errors closing stale connection
        except Exception:
            pass
        
        # Establish new connection
        self._connect()
        logger.info("TursoStore: Reconnection successful")
    
    def _execute_with_retry(self, operation: Callable[[], T], operation_name: str = "operation") -> T:
        """
        Execute a database operation with automatic retry on connection errors.
        
        Uses infinite retry with exponential backoff for transient connection issues.
        This ensures long-running ingestion jobs don't fail due to temporary network hiccups.
        
        Args:
            operation: Callable that performs the database operation
            operation_name: Name of operation for logging
            
        Returns:
            Result of the operation
            
        Raises:
            Non-connection errors immediately (don't retry logic/data errors)
        """
        attempt = 0
        
        while True:
            try:
                return operation()
            except Exception as e:
                if _is_stream_error(e):
                    attempt += 1
                    
                    # Calculate delay with exponential backoff, capped at max
                    delay = min(
                        self._base_retry_delay * (2 ** (attempt - 1)),
                        self._max_retry_delay
                    )
                    
                    logger.warning(
                        f"TursoStore: Stream error during {operation_name} "
                        f"(attempt {attempt}, retrying in {delay:.1f}s): {e}"
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
                    
                    # Reconnect
                    self._reconnect()
                else:
                    # Not a connection error - don't retry, raise immediately
                    raise
    
    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Entities table (nodes) with native vector support
        # F32_BLOB(n) enables native vector similarity search
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                user_id TEXT,
                name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                accessed_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                embedding F32_BLOB({self.embedding_dimensions}),
                metadata TEXT
            )
        """)
        
        # Create native vector index for fast similarity search
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS entities_vector_idx 
                ON entities (libsql_vector_idx(embedding))
            """)
            self._native_vector_search = True
            logger.info("Native vector search index created")
        except Exception as e:
            # Fallback to Python-based search if vector index fails
            self._native_vector_search = False
            logger.warning(f"Native vector index not available, using Python fallback: {e}")
        
        # Relationships table (edges)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                user_id TEXT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                description TEXT,
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 1.0,
                valid_from TEXT,
                valid_until TEXT,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        """)
        
        # Clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER,
                memory_id TEXT NOT NULL,
                user_id TEXT,
                summary TEXT,
                entity_ids TEXT,
                metadata TEXT,
                PRIMARY KEY (id, memory_id, user_id)
            )
        """)
        
        # Memory metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """)
        
        # Create indices for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_memory_user 
            ON entities(memory_id, user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relationships_memory_user 
            ON relationships(memory_id, user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name 
            ON entities(name)
        """)
        
        self.conn.commit()
        logger.debug("TursoStore schema initialized")
    
    def save_memory(self, memory: Memory) -> None:
        """Save memory to Turso database with automatic retry on connection errors."""
        def _do_save():
            self._save_memory_internal(memory)
        
        self._execute_with_retry(_do_save, "save_memory")
    
    def _save_memory_internal(self, memory: Memory) -> None:
        """Internal save implementation."""
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        
        # Save/update memory metadata
        memory_created_at = now
        if hasattr(memory, 'created_at') and memory.created_at:
            memory_created_at = memory.created_at.isoformat() if isinstance(memory.created_at, datetime) else str(memory.created_at)
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories (id, user_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            memory.id,
            getattr(memory, 'user_id', None),
            memory_created_at,
            now,
            json.dumps(getattr(memory, 'metadata', {}) or {})
        ))
        
        # Save nodes (use dict.copy() to safely avoid concurrent modification)
        nodes_snapshot = list(memory.nodes.copy().items())
        for node_id, node in nodes_snapshot:
            # Convert importance to float/int if it's an enum
            importance_val = node.importance
            if hasattr(importance_val, 'value'):
                importance_val = importance_val.value
            
            # Convert datetime objects to ISO strings
            accessed_at_str = None
            if hasattr(node, 'accessed_at') and node.accessed_at:
                accessed_at_str = node.accessed_at.isoformat() if isinstance(node.accessed_at, datetime) else str(node.accessed_at)
            
            created_at_str = now
            if hasattr(node, 'created_at') and node.created_at:
                created_at_str = node.created_at.isoformat() if isinstance(node.created_at, datetime) else str(node.created_at)
            
            metadata_json = json.dumps(node.properties) if hasattr(node, 'properties') and node.properties else None
            
            # Use native vector format if embedding exists
            if node.embedding and getattr(self, '_native_vector_search', False):
                # Format embedding as string for vector32() function
                embedding_str = str(list(node.embedding))
                cursor.execute(f"""
                    INSERT OR REPLACE INTO entities 
                    (id, memory_id, user_id, name, entity_type, description, 
                     importance, access_count, accessed_at, created_at, updated_at,
                     embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, vector32('{embedding_str}'), ?)
                """, (
                    node_id,
                    memory.id,
                    node.user_id,
                    node.name,
                    node.entity_type,
                    node.description,
                    importance_val,
                    node.access_count,
                    accessed_at_str,
                    created_at_str,
                    now,
                    metadata_json
                ))
            else:
                # Fallback: store embedding as JSON string
                embedding_json = json.dumps(node.embedding) if node.embedding else None
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, memory_id, user_id, name, entity_type, description, 
                     importance, access_count, accessed_at, created_at, updated_at,
                     embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node_id,
                    memory.id,
                    node.user_id,
                    node.name,
                    node.entity_type,
                    node.description,
                    importance_val,
                    node.access_count,
                    accessed_at_str,
                    created_at_str,
                    now,
                    embedding_json,
                    metadata_json
                ))
        
        # Save edges (use dict.copy() to safely avoid concurrent modification)
        edges_snapshot = list(memory.edges.copy().items())
        for edge_id, edge in edges_snapshot:
            # Convert datetime objects to ISO strings
            valid_from_str = None
            if hasattr(edge, 'valid_from') and edge.valid_from:
                valid_from_str = edge.valid_from.isoformat() if isinstance(edge.valid_from, datetime) else str(edge.valid_from)
            
            valid_until_str = None
            if hasattr(edge, 'valid_until') and edge.valid_until:
                valid_until_str = edge.valid_until.isoformat() if isinstance(edge.valid_until, datetime) else str(edge.valid_until)
            
            created_at_str = now
            if hasattr(edge, 'created_at') and edge.created_at:
                created_at_str = edge.created_at.isoformat() if isinstance(edge.created_at, datetime) else str(edge.created_at)
            
            cursor.execute("""
                INSERT OR REPLACE INTO relationships
                (id, memory_id, user_id, source_id, target_id, relation_type,
                 description, weight, confidence, valid_from, valid_until,
                 created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge_id,
                memory.id,
                getattr(memory, 'user_id', None),  # Use memory's user_id since edges don't have one
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                getattr(edge, 'description', None),
                getattr(edge, 'weight', 1.0),
                getattr(edge, 'confidence', 1.0),
                valid_from_str,
                valid_until_str,
                created_at_str,
                json.dumps(edge.properties) if hasattr(edge, 'properties') and edge.properties else None
            ))
        
        # Save clusters (use dict.copy() to safely avoid concurrent modification)
        clusters_snapshot = list(memory.clusters.copy().items())
        for cluster_id, cluster in clusters_snapshot:
            cursor.execute("""
                INSERT OR REPLACE INTO clusters
                (id, memory_id, user_id, summary, entity_ids, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                cluster_id,
                memory.id,
                getattr(memory, 'user_id', None),
                cluster.summary,
                json.dumps(list(cluster.entities if hasattr(cluster, 'entities') else cluster.entity_ids)),
                json.dumps(cluster.metadata) if cluster.metadata else None
            ))
        
        self.conn.commit()
        
        # Sync to cloud if configured
        if self.turso_url and self.sync_mode in ("full", "push"):
            self._sync()
        
        logger.debug(f"Saved memory {memory.id} with {len(memory.nodes)} nodes, {len(memory.edges)} edges")
    
    def load_memory(self, memory_id: str, user_id: str = None) -> Optional[Memory]:
        """
        Load memory from Turso database with automatic retry on connection errors.
        
        Args:
            memory_id: Memory ID to load
            user_id: User ID for multi-tenant isolation
        """
        def _do_load():
            return self._load_memory_internal(memory_id, user_id)
        
        return self._execute_with_retry(_do_load, "load_memory")
    
    def _load_memory_internal(self, memory_id: str, user_id: str = None) -> Optional[Memory]:
        """Internal load implementation."""
        # Sync from cloud first if configured
        if self.turso_url and self.sync_mode in ("full", "pull"):
            self._sync()
        
        cursor = self.conn.cursor()
        
        # Check if memory exists
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        memory_row = cursor.fetchone()
        if not memory_row:
            return None
        
        # Load nodes
        if user_id:
            cursor.execute("""
                SELECT * FROM entities 
                WHERE memory_id = ? AND (user_id = ? OR user_id IS NULL)
            """, (memory_id, user_id))
        else:
            cursor.execute("SELECT * FROM entities WHERE memory_id = ?", (memory_id,))
        
        nodes = {}
        for row in cursor.fetchall():
            # Unpack embedding if present
            embedding = None
            if row[11]:  # embedding blob
                import struct
                embedding_blob = row[11]
                num_floats = len(embedding_blob) // 4
                embedding = list(struct.unpack(f'{num_floats}f', embedding_blob))
            
            # Convert raw importance value to MemoryImportance enum
            importance_val = row[6] if row[6] is not None else 5
            try:
                importance = MemoryImportance(int(importance_val))
            except (ValueError, TypeError):
                importance = MemoryImportance.MEDIUM
            
            node = MemoryNode(
                id=row[0],
                name=row[3],
                entity_type=row[4],
                description=row[5],
                importance=importance,
                access_count=row[7] or 0,
                accessed_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.utcnow(),
                embedding=embedding,
                properties=json.loads(row[12]) if row[12] else {},
                user_id=row[2],
            )
            nodes[row[0]] = node
        
        # Load edges (only between loaded nodes)
        node_ids = list(nodes.keys())
        if not node_ids:
            placeholders = "()"
        else:
            placeholders = f"({','.join(['?'] * len(node_ids))})"
        
        if user_id:
            cursor.execute(f"""
                SELECT * FROM relationships 
                WHERE memory_id = ? 
                AND source_id IN {placeholders}
                AND target_id IN {placeholders}
                AND (user_id = ? OR user_id IS NULL)
            """, (memory_id, *node_ids, *node_ids, user_id))
        else:
            if node_ids:
                cursor.execute(f"""
                    SELECT * FROM relationships 
                    WHERE memory_id = ? 
                    AND source_id IN {placeholders}
                    AND target_id IN {placeholders}
                """, (memory_id, *node_ids, *node_ids))
            else:
                cursor.execute("SELECT * FROM relationships WHERE memory_id = ?", (memory_id,))
        
        edges = {}
        for row in cursor.fetchall():
            edge = MemoryEdge(
                id=row[0],
                source_id=row[3],
                target_id=row[4],
                relation_type=row[5],
                description=row[6],
                weight=row[7] or 1.0,
                confidence=row[8] or 1.0,
                valid_from=datetime.fromisoformat(row[9]) if row[9] else None,
                valid_until=datetime.fromisoformat(row[10]) if row[10] else None,
                created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
                properties=json.loads(row[12]) if row[12] else {},
            )
            edges[row[0]] = edge
        
        # Load clusters
        cursor.execute("""
            SELECT * FROM clusters WHERE memory_id = ?
        """, (memory_id,))
        
        clusters = {}
        for row in cursor.fetchall():
            cluster = MemoryCluster(
                id=row[0],
                entities=json.loads(row[4]) if row[4] else [],
                summary=row[3],
            )
            clusters[row[0]] = cluster
        
        # Create Memory object
        memory = Memory(
            id=memory_id,
            nodes=nodes,
            edges=edges,
            clusters=clusters,
        )
        
        logger.debug(f"Loaded memory {memory_id}: {len(nodes)} nodes, {len(edges)} edges")
        return memory
    
    def delete_memory(self, memory_id: str) -> None:
        """Delete memory from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM entities WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM relationships WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM clusters WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        
        if self.turso_url and self.sync_mode in ("full", "push"):
            self._sync()
        
        logger.debug(f"Deleted memory {memory_id}")
    
    def clear_memory(self, memory_id: str) -> None:
        """Clear all data in a memory (same as delete)."""
        self.delete_memory(memory_id)
    
    def list_memories(self) -> List[str]:
        """List all memory IDs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM memories")
        return [row[0] for row in cursor.fetchall()]
    
    def vector_search(
        self,
        memory_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
    ) -> List[MemoryNode]:
        """
        Perform vector similarity search using native libsql vector_top_k().
        
        Uses libsql's native F32_BLOB vector search for optimal performance.
        Falls back to Python-based cosine similarity if native search unavailable.
        
        Args:
            memory_id: Memory to search in
            query_embedding: Query vector
            top_k: Number of results to return
            user_id: Filter by user
            
        Returns:
            List of most similar nodes
        """
        cursor = self.conn.cursor()
        
        # Try native vector search first
        if getattr(self, '_native_vector_search', False):
            try:
                # Format query embedding for vector_top_k
                query_str = str(list(query_embedding))
                
                # Native vector search with vector_top_k
                # Use subquery to avoid ambiguous column name
                results = cursor.execute(f"""
                    SELECT e.id, e.memory_id, e.user_id, 
                           e.name, e.entity_type, e.description,
                           e.importance, e.access_count, e.accessed_at,
                           e.created_at, e.updated_at, e.metadata
                    FROM vector_top_k('entities_vector_idx', '{query_str}', {top_k * 3}) AS v
                    JOIN entities AS e ON e.rowid = v.id
                    WHERE e.memory_id = ?
                    AND (e.user_id = ? OR e.user_id IS NULL OR ? IS NULL)
                    LIMIT ?
                """, (memory_id, user_id, user_id, top_k)).fetchall()
                
                nodes = []
                for row in results:
                    node = MemoryNode(
                        id=row[0],
                        name=row[3],
                        entity_type=row[4],
                        description=row[5],
                        importance=row[6] or 0.5,
                        access_count=row[7] or 0,
                        accessed_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                        created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.utcnow(),
                        embedding=query_embedding,  # Don't need actual embedding in results
                        properties=json.loads(row[11]) if row[11] else {},
                        user_id=row[2],
                    )
                    nodes.append(node)
                
                logger.debug(f"Native vector search returned {len(nodes)} results")
                return nodes
                
            except Exception as e:
                logger.warning(f"Native vector search failed, falling back to Python: {e}")
        
        # Fallback: Python-based cosine similarity
        return self._vector_search_fallback(memory_id, query_embedding, top_k, user_id)
    
    def _vector_search_fallback(
        self,
        memory_id: str,
        query_embedding: List[float],
        top_k: int,
        user_id: Optional[str],
    ) -> List[MemoryNode]:
        """Fallback vector search using Python cosine similarity."""
        import math
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        
        cursor = self.conn.cursor()
        
        # Load all nodes with embeddings (stored as JSON in fallback mode)
        if user_id:
            cursor.execute("""
                SELECT id, embedding FROM entities 
                WHERE memory_id = ? AND embedding IS NOT NULL
                AND (user_id = ? OR user_id IS NULL)
            """, (memory_id, user_id))
        else:
            cursor.execute("""
                SELECT id, embedding FROM entities 
                WHERE memory_id = ? AND embedding IS NOT NULL
            """, (memory_id,))
        
        # Calculate similarities
        similarities = []
        for row in cursor.fetchall():
            node_id = row[0]
            embedding_data = row[1]
            
            # Parse embedding (could be JSON string or binary)
            try:
                if isinstance(embedding_data, str):
                    node_embedding = json.loads(embedding_data)
                elif isinstance(embedding_data, bytes):
                    import struct
                    num_floats = len(embedding_data) // 4
                    node_embedding = list(struct.unpack(f'{num_floats}f', embedding_data))
                else:
                    continue
                    
                sim = cosine_similarity(query_embedding, node_embedding)
                similarities.append((node_id, sim))
            except Exception:
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ids = [node_id for node_id, _ in similarities[:top_k]]
        
        # Load full nodes
        result = []
        for node_id in top_ids:
            cursor.execute("SELECT * FROM entities WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                node = MemoryNode(
                    id=row[0],
                    name=row[3],
                    entity_type=row[4],
                    description=row[5],
                    importance=row[6] or 0.5,
                    access_count=row[7] or 0,
                    accessed_at=datetime.fromisoformat(row[8]) if row[8] else datetime.utcnow(),
                    created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.utcnow(),
                    embedding=query_embedding,
                    properties=json.loads(row[12]) if row[12] else {},
                    user_id=row[2],
                )
                result.append(node)
        
        return result
    
    def query_edges_at_time(
        self,
        memory_id: str,
        query_time: datetime,
        relation_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[MemoryEdge]:
        """
        Query edges that are valid at a specific point in time.
        
        Args:
            memory_id: Memory to query
            query_time: The point in time to query
            relation_type: Optional filter by relationship type
            user_id: Optional filter by user
            
        Returns:
            List of edges valid at the specified time
        """
        cursor = self.conn.cursor()
        query_time_str = query_time.isoformat()
        
        query = """
            SELECT * FROM relationships
            WHERE memory_id = ?
            AND (valid_from IS NULL OR valid_from <= ?)
            AND (valid_until IS NULL OR valid_until > ?)
        """
        params = [memory_id, query_time_str, query_time_str]
        
        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
        
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        
        cursor.execute(query, params)
        
        edges = []
        for row in cursor.fetchall():
            edge = MemoryEdge(
                id=row[0],
                source_id=row[3],
                target_id=row[4],
                relation_type=row[5],
                description=row[6],
                weight=row[7] or 1.0,
                confidence=row[8] or 1.0,
                valid_from=datetime.fromisoformat(row[9]) if row[9] else None,
                valid_until=datetime.fromisoformat(row[10]) if row[10] else None,
                created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
                properties=json.loads(row[12]) if row[12] else {},
            )
            edges.append(edge)
        
        return edges
    
    def supersede_relationship(
        self,
        memory_id: str,
        edge_id: str,
        end_time: Optional[datetime] = None,
    ) -> bool:
        """
        Mark a relationship as superseded (no longer valid).
        
        Used when facts change, e.g., CEO transitions.
        The old relationship is preserved for historical queries.
        
        Args:
            memory_id: Memory containing the edge
            edge_id: ID of edge to supersede
            end_time: When the relationship ended (defaults to now)
        
        Returns:
            True if successful
        """
        end_time = end_time or datetime.utcnow()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE relationships 
            SET valid_until = ?, state = 'ARCHIVED'
            WHERE id = ? AND memory_id = ?
        """, (end_time.isoformat(), edge_id, memory_id))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def _sync(self) -> None:
        """Sync with Turso Cloud."""
        try:
            if hasattr(self.conn, 'sync'):
                self.conn.sync()
                logger.debug("Synced with Turso Cloud")
        except Exception as e:
            logger.warning(f"Turso sync failed: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.turso_url and self.sync_mode in ("full", "push"):
            self._sync()
        self.conn.close()
        logger.info("TursoStore connection closed")
    
    def health_check(self) -> bool:
        """Check if store is healthy."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False


class TursoCache:
    """
    SQLite-based cache using Turso/libSQL.
    
    Alternative to Redis cache with:
    - Persistence (survives restarts)
    - No external server needed
    - Offline support
    - Optional cloud sync
    """
    
    def __init__(
        self,
        db_path: str = "graphmem_cache.db",
        ttl: int = 3600,
        turso_url: Optional[str] = None,
        turso_auth_token: Optional[str] = None,
    ):
        """Initialize Turso cache."""
        if not TURSO_AVAILABLE:
            raise ImportError(
                "libsql_experimental is required for TursoCache. "
                "Install with: pip install libsql-experimental"
            )
        
        self.ttl = ttl
        
        if turso_url and turso_auth_token:
            self.conn = libsql.connect(
                db_path,
                sync_url=turso_url,
                auth_token=turso_auth_token,
            )
        else:
            self.conn = libsql.connect(db_path)
        
        self._init_schema()
        logger.info(f"TursoCache initialized: {db_path}")
    
    def _init_schema(self) -> None:
        """Create cache table."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
        self.conn.commit()
    
    def _key(self, *parts: str) -> str:
        """Build a cache key."""
        return ":".join(str(p) for p in parts)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        self.conn.commit()
    
    def get(self, *key_parts: str) -> Optional[Any]:
        """Get value from cache."""
        key = self._key(*key_parts) if len(key_parts) > 1 else key_parts[0]
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            SELECT value FROM cache 
            WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
        """, (key, now))
        
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def set(self, *key_parts: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        key = self._key(*key_parts)
        ttl = ttl or self.ttl
        expires_at = None
        if ttl:
            from datetime import timedelta
            expires_at = (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO cache (key, value, expires_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), expires_at))
        self.conn.commit()
        return True
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        self.conn.commit()
    
    def invalidate(self, memory_id: str, user_id: str = "default") -> None:
        """Invalidate all cache entries for a user's memory."""
        cursor = self.conn.cursor()
        pattern = f"%{user_id}%{memory_id}%"
        cursor.execute("DELETE FROM cache WHERE key LIKE ?", (pattern,))
        deleted = cursor.rowcount
        self.conn.commit()
        logger.debug(f"Invalidated {deleted} Turso cache entries")
    
    # Specialized cache methods matching InMemoryCache interface
    
    def get_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self.get(f"embedding:{text_hash}")
    
    def cache_embedding(self, text_hash: str, embedding: List[float], ttl: int = 86400) -> bool:
        """Cache an embedding."""
        return self.set(f"embedding:{text_hash}", value=embedding, ttl=ttl)
    
    def get_search_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[list]:
        """Get cached search results."""
        return self.get("search", user_id, memory_id, query_hash)
    
    def cache_search_result(
        self,
        memory_id: str,
        query_hash: str,
        results: list,
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """Cache search results."""
        return self.set("search", user_id, memory_id, query_hash, value=results, ttl=ttl)
    
    def get_query_result(
        self,
        memory_id: str,
        query_hash: str,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        return self.get("query", user_id, memory_id, query_hash)
    
    def cache_query_result(
        self,
        memory_id: str,
        query_hash: str,
        result: Dict[str, Any],
        user_id: str = "default",
        ttl: int = 300,
    ) -> bool:
        """Cache query result."""
        return self.set("query", user_id, memory_id, query_hash, value=result, ttl=ttl)
    
    def get_community_context(
        self,
        memory_id: str,
        community_id: int,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get cached community context."""
        return self.get("community", user_id, memory_id, str(community_id))
    
    def cache_community_context(
        self,
        memory_id: str,
        community_id: int,
        context: Dict[str, Any],
        user_id: str = "default",
    ) -> bool:
        """Cache community context."""
        return self.set("community", user_id, memory_id, str(community_id), value=context)
    
    def close(self) -> None:
        """Close cache connection."""
        self.conn.close()


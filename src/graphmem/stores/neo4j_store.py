"""
GraphMem Neo4j Store

Production-grade Neo4j backend for persistent graph storage.
Includes retry logic, connection pooling, optimized queries,
and native vector index support for fast semantic search.
"""

from __future__ import annotations
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from graphmem.core.memory_types import Memory, MemoryNode, MemoryEdge, MemoryCluster, MemoryImportance
from graphmem.core.exceptions import StorageError

logger = logging.getLogger(__name__)

# Default embedding dimensions for common models
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Neo4jStore:
    """
    Neo4j storage backend for GraphMem.
    
    Features:
    - Automatic retry on transient failures
    - Connection pooling
    - Optimized batch operations
    - Full ACID compliance
    """
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        embedding_dimensions: int = 1536,
        use_vector_index: bool = True,
    ):
        """
        Initialize Neo4j store.
        
        Args:
            uri: Neo4j connection URI (bolt://...)
            username: Database username
            password: Database password
            database: Database name
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            embedding_dimensions: Dimension of embedding vectors (default 1536 for OpenAI)
            use_vector_index: Whether to use Neo4j vector index for fast similarity search
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.embedding_dimensions = embedding_dimensions
        self.use_vector_index = use_vector_index
        self._driver = None
        self._vector_index_created = False
    
    @property
    def driver(self):
        """Lazy initialization of Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                )
            except ImportError:
                raise ImportError("neo4j package required: pip install neo4j")
        return self._driver
    
    def _execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        write: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a query with retry logic."""
        params = params or {}
        
        for attempt in range(self.max_retries):
            try:
                with self.driver.session(database=self.database) as session:
                    if write:
                        result = session.execute_write(
                            lambda tx: list(tx.run(query, params))
                        )
                    else:
                        result = session.execute_read(
                            lambda tx: list(tx.run(query, params))
                        )
                    return [dict(record) for record in result]
                    
            except Exception as e:
                logger.warning(f"Neo4j query failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise StorageError(
                        f"Neo4j query failed after {self.max_retries} attempts: {e}",
                        storage_type="neo4j",
                        operation="query",
                        cause=e,
                    )
        return []
    
    def save_memory(self, memory: Memory) -> None:
        """Save a memory to Neo4j."""
        # Save memory metadata
        self._execute_query(
            """
            MERGE (m:Memory {id: $id})
            SET m.name = $name,
                m.description = $description,
                m.importance = $importance,
                m.state = $state,
                m.version = $version,
                m.created_at = $created_at,
                m.updated_at = $updated_at
            """,
            {
                "id": memory.id,
                "name": memory.name,
                "description": memory.description,
                "importance": memory.importance.value,
                "state": memory.state.name,
                "version": memory.version,
                "created_at": memory.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
            write=True,
        )
        
        # Save nodes in batches
        self._save_nodes_batch(memory.id, list(memory.nodes.values()))
        
        # Save edges in batches
        self._save_edges_batch(memory.id, list(memory.edges.values()))
        
        # Save clusters
        self._save_clusters(memory.id, list(memory.clusters.values()))
        
        logger.info(f"Saved memory {memory.id}: {len(memory.nodes)} nodes, {len(memory.edges)} edges")
    
    def _save_nodes_batch(self, memory_id: str, nodes: List[MemoryNode], batch_size: int = 500) -> None:
        """Save nodes in batches, including embeddings for vector search."""
        if not nodes:
            return
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            node_data = [
                {
                    "id": n.id,
                    "name": n.name,
                    "entity_type": n.entity_type,
                    "description": n.description,
                    "canonical_name": n.canonical_name,
                    "aliases": list(n.aliases),
                    "properties": json.dumps(n.properties),
                    "importance": n.importance.value,
                    "state": n.state.name,
                    "access_count": n.access_count,
                    "created_at": n.created_at.isoformat(),
                    "updated_at": n.updated_at.isoformat(),
                    "accessed_at": n.accessed_at.isoformat(),
                    "embedding": n.embedding,  # Include embedding for vector search
                }
                for n in batch
            ]
            
            self._execute_query(
                """
                UNWIND $nodes AS node
                MERGE (n:Entity {id: node.id, memory_id: $memory_id})
                SET n.name = node.name,
                    n.entity_type = node.entity_type,
                    n.description = node.description,
                    n.canonical_name = node.canonical_name,
                    n.aliases = node.aliases,
                    n.properties = node.properties,
                    n.importance = node.importance,
                    n.state = node.state,
                    n.access_count = node.access_count,
                    n.created_at = node.created_at,
                    n.updated_at = node.updated_at,
                    n.accessed_at = node.accessed_at,
                    n.embedding = node.embedding
                """,
                {"memory_id": memory_id, "nodes": node_data},
                write=True,
            )
        
        # Ensure vector index exists after saving nodes with embeddings
        if self.use_vector_index:
            self.ensure_vector_index(memory_id)
    
    def _save_edges_batch(self, memory_id: str, edges: List[MemoryEdge], batch_size: int = 500) -> None:
        """Save edges in batches."""
        if not edges:
            return
        
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            edge_data = [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation_type": e.relation_type,
                    "description": e.description,
                    "weight": e.weight,
                    "confidence": e.confidence,
                    "properties": json.dumps(e.properties),
                    "importance": e.importance.value,
                    "state": e.state.name,
                }
                for e in batch
            ]
            
            self._execute_query(
                """
                UNWIND $edges AS edge
                MATCH (s:Entity {id: edge.source_id, memory_id: $memory_id})
                MATCH (t:Entity {id: edge.target_id, memory_id: $memory_id})
                MERGE (s)-[r:RELATED {id: edge.id}]->(t)
                SET r.relation_type = edge.relation_type,
                    r.description = edge.description,
                    r.weight = edge.weight,
                    r.confidence = edge.confidence,
                    r.properties = edge.properties,
                    r.importance = edge.importance,
                    r.state = edge.state,
                    r.memory_id = $memory_id
                """,
                {"memory_id": memory_id, "edges": edge_data},
                write=True,
            )
    
    def _save_clusters(self, memory_id: str, clusters: List[MemoryCluster]) -> None:
        """Save clusters."""
        if not clusters:
            return
        
        for cluster in clusters:
            self._execute_query(
                """
                MERGE (c:Community {id: $id, memory_id: $memory_id})
                SET c.summary = $summary,
                    c.entities = $entities,
                    c.importance = $importance,
                    c.coherence_score = $coherence_score,
                    c.density = $density,
                    c.updated_at = $updated_at
                """,
                {
                    "memory_id": memory_id,
                    "id": cluster.id,
                    "summary": cluster.summary,
                    "entities": cluster.entities,
                    "importance": cluster.importance.value,
                    "coherence_score": cluster.coherence_score,
                    "density": cluster.density,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                write=True,
            )
    
    def load_memory(self, memory_id: str) -> Optional[Memory]:
        """Load a memory from Neo4j."""
        # Load memory metadata
        result = self._execute_query(
            """
            MATCH (m:Memory {id: $id})
            RETURN m
            """,
            {"id": memory_id},
        )
        
        if not result:
            return None
        
        memory_data = result[0]["m"]
        
        # Create memory object
        from graphmem.core.memory_types import MemoryImportance, MemoryState
        
        memory = Memory(
            id=memory_data.get("id", memory_id),
            name=memory_data.get("name"),
            description=memory_data.get("description"),
            importance=MemoryImportance(memory_data.get("importance", 5)),
            state=MemoryState[memory_data.get("state", "ACTIVE")],
            version=memory_data.get("version", 1),
        )
        
        # Load nodes
        nodes = self._load_nodes(memory_id)
        for node in nodes:
            memory.nodes[node.id] = node
        
        # Load edges
        edges = self._load_edges(memory_id)
        for edge in edges:
            memory.edges[edge.id] = edge
        
        # Load clusters
        clusters = self._load_clusters(memory_id)
        for cluster in clusters:
            memory.clusters[cluster.id] = cluster
        
        logger.info(f"Loaded memory {memory_id}: {len(memory.nodes)} nodes")
        return memory
    
    def _load_nodes(self, memory_id: str) -> List[MemoryNode]:
        """Load nodes for a memory, including embeddings."""
        result = self._execute_query(
            """
            MATCH (n:Entity {memory_id: $memory_id})
            RETURN n
            """,
            {"memory_id": memory_id},
        )
        
        nodes = []
        for record in result:
            n = record["n"]
            try:
                props = json.loads(n.get("properties", "{}"))
            except:
                props = {}
            
            from graphmem.core.memory_types import MemoryImportance, MemoryState
            
            node = MemoryNode(
                id=n["id"],
                name=n["name"],
                entity_type=n.get("entity_type", "Entity"),
                description=n.get("description"),
                canonical_name=n.get("canonical_name"),
                aliases=set(n.get("aliases", [])),
                properties=props,
                embedding=n.get("embedding"),  # Load embedding for vector search
                importance=MemoryImportance(n.get("importance", 5)),
                state=MemoryState[n.get("state", "ACTIVE")],
                access_count=n.get("access_count", 0),
                memory_id=memory_id,
            )
            nodes.append(node)
        
        return nodes
    
    def _load_edges(self, memory_id: str) -> List[MemoryEdge]:
        """Load edges for a memory."""
        result = self._execute_query(
            """
            MATCH (s:Entity {memory_id: $memory_id})-[r:RELATED {memory_id: $memory_id}]->(t:Entity {memory_id: $memory_id})
            RETURN r, s.id AS source_id, t.id AS target_id
            """,
            {"memory_id": memory_id},
        )
        
        edges = []
        for record in result:
            r = record["r"]
            try:
                props = json.loads(r.get("properties", "{}"))
            except:
                props = {}
            
            from graphmem.core.memory_types import MemoryImportance, MemoryState
            
            edge = MemoryEdge(
                id=r["id"],
                source_id=record["source_id"],
                target_id=record["target_id"],
                relation_type=r.get("relation_type", "RELATED"),
                description=r.get("description"),
                weight=r.get("weight", 1.0),
                confidence=r.get("confidence", 1.0),
                properties=props,
                importance=MemoryImportance(r.get("importance", 5)),
                state=MemoryState[r.get("state", "ACTIVE")],
                memory_id=memory_id,
            )
            edges.append(edge)
        
        return edges
    
    def _load_clusters(self, memory_id: str) -> List[MemoryCluster]:
        """Load clusters for a memory."""
        result = self._execute_query(
            """
            MATCH (c:Community {memory_id: $memory_id})
            RETURN c
            """,
            {"memory_id": memory_id},
        )
        
        clusters = []
        for record in result:
            c = record["c"]
            
            from graphmem.core.memory_types import MemoryImportance
            
            cluster = MemoryCluster(
                id=c["id"],
                summary=c.get("summary", ""),
                entities=c.get("entities", []),
                importance=MemoryImportance(c.get("importance", 5)),
                coherence_score=c.get("coherence_score", 1.0),
                density=c.get("density", 1.0),
                memory_id=memory_id,
            )
            clusters.append(cluster)
        
        return clusters
    
    def clear_memory(self, memory_id: str) -> None:
        """Clear all data for a memory."""
        self._execute_query(
            """
            MATCH (n {memory_id: $memory_id})
            DETACH DELETE n
            """,
            {"memory_id": memory_id},
            write=True,
        )
        
        self._execute_query(
            """
            MATCH (m:Memory {id: $memory_id})
            DELETE m
            """,
            {"memory_id": memory_id},
            write=True,
        )
        
        logger.info(f"Cleared memory {memory_id}")
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    # ==================== Vector Index Support ====================
    
    def ensure_vector_index(self, memory_id: str = None) -> bool:
        """
        Ensure a vector index exists for Entity.embedding.
        
        Neo4j allows only ONE vector index per label-property combination.
        This method finds any existing vector index on Entity.embedding or creates one.
        
        Returns True if index exists/was created, False if not supported.
        """
        if not self.use_vector_index:
            return False
        
        if self._vector_index_created:
            return True
        
        try:
            # Check Neo4j version - vector indexes require 5.x+
            version_result = self._execute_query("CALL dbms.components() YIELD versions RETURN versions[0] AS version")
            if version_result:
                version = version_result[0].get("version", "0")
                major_version = int(version.split(".")[0])
                if major_version < 5:
                    logger.warning(f"Neo4j {version} does not support vector indexes. Need 5.x+")
                    self.use_vector_index = False
                    return False
            
            # Check for ANY existing vector index on Entity.embedding
            # Neo4j only allows ONE vector index per label-property pair
            existing = self._execute_query("""
                SHOW INDEXES 
                WHERE type = 'VECTOR' 
                AND entityType = 'NODE'
            """)
            
            for idx in existing:
                # Check if this index is on Entity label and embedding property
                labels = idx.get("labelsOrTypes", [])
                props = idx.get("properties", [])
                if "Entity" in labels and "embedding" in props:
                    self._vector_index_name = idx.get("name")
                    state = idx.get("state", "UNKNOWN")
                    if state == "ONLINE":
                        logger.info(f"Found existing vector index: {self._vector_index_name} (ONLINE)")
                        self._vector_index_created = True
                        return True
                    else:
                        logger.info(f"Found vector index {self._vector_index_name} in state: {state}")
                        self._vector_index_created = True
                        return True
            
            # No existing index - create one
            index_name = "graphmem_entity_vector_idx"
            self._vector_index_name = index_name
            
            self._execute_query(
                f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:Entity)
                ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {self.embedding_dimensions},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """,
                write=True,
            )
            
            logger.info(f"Created vector index {index_name}")
            
            # Wait for index to come ONLINE (up to 10 seconds)
            import time
            for _ in range(10):
                status = self._execute_query(
                    "SHOW INDEXES WHERE name = $name",
                    {"name": index_name}
                )
                if status and status[0].get("state") == "ONLINE":
                    logger.info(f"Vector index {index_name} is ONLINE")
                    self._vector_index_created = True
                    return True
                time.sleep(1)
            
            self._vector_index_created = True
            return True
            
        except Exception as e:
            logger.warning(f"Could not ensure vector index: {e}. Falling back to in-memory search.")
            self.use_vector_index = False
            return False
    
    def save_embedding(self, memory_id: str, node_id: str, embedding: List[float]) -> None:
        """Save embedding vector for a node."""
        self._execute_query(
            """
            MATCH (n:Entity {id: $node_id, memory_id: $memory_id})
            SET n.embedding = $embedding
            """,
            {
                "memory_id": memory_id,
                "node_id": node_id,
                "embedding": embedding,
            },
            write=True,
        )
    
    def save_embeddings_batch(
        self, 
        memory_id: str, 
        embeddings: Dict[str, List[float]], 
        batch_size: int = 100
    ) -> None:
        """
        Save multiple embeddings in batches.
        
        Args:
            memory_id: Memory ID
            embeddings: Dict mapping node_id -> embedding vector
            batch_size: Number of embeddings per batch
        """
        items = list(embeddings.items())
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_data = [{"node_id": nid, "embedding": emb} for nid, emb in batch]
            
            self._execute_query(
                """
                UNWIND $batch AS item
                MATCH (n:Entity {id: item.node_id, memory_id: $memory_id})
                SET n.embedding = item.embedding
                """,
                {"memory_id": memory_id, "batch": batch_data},
                write=True,
            )
        
        logger.info(f"Saved {len(embeddings)} embeddings for memory {memory_id}")
    
    def vector_search(
        self,
        memory_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Perform vector similarity search using Neo4j vector index.
        
        Args:
            memory_id: Memory ID to search in
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (MemoryNode, similarity_score) tuples
        """
        if not self.use_vector_index:
            return self._vector_search_fallback(memory_id, query_embedding, top_k, min_score)
        
        # Ensure index exists
        self.ensure_vector_index()
        
        if not self._vector_index_created or not hasattr(self, '_vector_index_name'):
            return self._vector_search_fallback(memory_id, query_embedding, top_k, min_score)
        
        # Use the discovered/created index name
        index_name = self._vector_index_name
        
        try:
            # Use Neo4j vector search
            results = self._execute_query(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', $top_k, $embedding)
                YIELD node, score
                WHERE node.memory_id = $memory_id AND score >= $min_score
                RETURN node, score
                ORDER BY score DESC
                """,
                {
                    "embedding": query_embedding,
                    "top_k": top_k * 2,  # Fetch more to filter by memory_id
                    "memory_id": memory_id,
                    "min_score": min_score,
                },
            )
            
            nodes_with_scores = []
            for record in results[:top_k]:
                n = record["node"]
                score = record["score"]
                
                try:
                    props = json.loads(n.get("properties", "{}"))
                except:
                    props = {}
                
                # Apply evolution weighting to score
                importance_value = n.get("importance", 5)
                importance_weight = importance_value / 10.0
                
                # Calculate recency boost
                recency_boost = 0.0
                accessed_at_str = n.get("accessed_at")
                if accessed_at_str:
                    try:
                        accessed_at = datetime.fromisoformat(accessed_at_str)
                        hours_since = (datetime.utcnow() - accessed_at).total_seconds() / 3600
                        if hours_since < 24:
                            recency_boost = 0.1 * (1 - hours_since / 24)
                    except:
                        pass
                
                # Access count boost
                access_count = n.get("access_count", 0)
                access_boost = min(0.1, access_count * 0.01)
                
                # Combined score (same formula as semantic_search.py)
                combined_score = (
                    0.60 * score + 
                    0.25 * importance_weight + 
                    recency_boost + 
                    access_boost
                )
                
                node = MemoryNode(
                    id=n["id"],
                    name=n["name"],
                    entity_type=n.get("entity_type", "Entity"),
                    description=n.get("description"),
                    canonical_name=n.get("canonical_name"),
                    aliases=set(n.get("aliases", [])),
                    properties=props,
                    importance=MemoryImportance(importance_value),
                    access_count=access_count,
                    memory_id=memory_id,
                )
                
                # Skip EPHEMERAL (fully decayed) nodes
                if node.importance.value == 0:
                    continue
                
                nodes_with_scores.append((node, combined_score))
            
            # Re-sort by combined score
            nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Vector search returned {len(nodes_with_scores)} results")
            return nodes_with_scores[:top_k]
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}. Using fallback.")
            return self._vector_search_fallback(memory_id, query_embedding, top_k, min_score)
    
    def _vector_search_fallback(
        self,
        memory_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Fallback brute-force vector search when vector index not available.
        """
        # Load all nodes with embeddings
        results = self._execute_query(
            """
            MATCH (n:Entity {memory_id: $memory_id})
            WHERE n.embedding IS NOT NULL
            RETURN n
            """,
            {"memory_id": memory_id},
        )
        
        import math
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        
        nodes_with_scores = []
        for record in results:
            n = record["n"]
            embedding = n.get("embedding")
            if not embedding:
                continue
            
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity < min_score:
                continue
            
            try:
                props = json.loads(n.get("properties", "{}"))
            except:
                props = {}
            
            importance_value = n.get("importance", 5)
            
            # Skip EPHEMERAL nodes
            if importance_value == 0:
                continue
            
            importance_weight = importance_value / 10.0
            access_count = n.get("access_count", 0)
            access_boost = min(0.1, access_count * 0.01)
            
            # Recency boost
            recency_boost = 0.0
            accessed_at_str = n.get("accessed_at")
            if accessed_at_str:
                try:
                    accessed_at = datetime.fromisoformat(accessed_at_str)
                    hours_since = (datetime.utcnow() - accessed_at).total_seconds() / 3600
                    if hours_since < 24:
                        recency_boost = 0.1 * (1 - hours_since / 24)
                except:
                    pass
            
            combined_score = (
                0.60 * similarity + 
                0.25 * importance_weight + 
                recency_boost + 
                access_boost
            )
            
            node = MemoryNode(
                id=n["id"],
                name=n["name"],
                entity_type=n.get("entity_type", "Entity"),
                description=n.get("description"),
                canonical_name=n.get("canonical_name"),
                aliases=set(n.get("aliases", [])),
                properties=props,
                importance=MemoryImportance(importance_value),
                access_count=access_count,
                memory_id=memory_id,
            )
            nodes_with_scores.append((node, combined_score))
        
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        return nodes_with_scores[:top_k]
    
    def has_vector_support(self) -> bool:
        """Check if Neo4j vector index is available and enabled."""
        return self.use_vector_index and self._vector_index_created


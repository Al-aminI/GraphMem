"""
GraphMem Neo4j Store

Production-grade Neo4j backend for persistent graph storage.
Includes retry logic, connection pooling, and optimized queries.
"""

from __future__ import annotations
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from graphmem.core.memory_types import Memory, MemoryNode, MemoryEdge, MemoryCluster
from graphmem.core.exceptions import StorageError

logger = logging.getLogger(__name__)


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
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._driver = None
    
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
        """Save nodes in batches."""
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
                    n.accessed_at = node.accessed_at
                """,
                {"memory_id": memory_id, "nodes": node_data},
                write=True,
            )
    
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
        """Load nodes for a memory."""
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


"""
GraphMem Graph Module

Knowledge graph construction, entity resolution, and community detection.
"""

from graphmem.graph.knowledge_graph import KnowledgeGraph
from graphmem.graph.entity_resolver import EntityResolver
from graphmem.graph.community_detector import CommunityDetector

__all__ = [
    "KnowledgeGraph",
    "EntityResolver",
    "CommunityDetector",
]


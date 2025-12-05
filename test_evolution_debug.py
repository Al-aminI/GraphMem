#!/usr/bin/env python3
"""
GraphMem Evolution Debug Test

Tests the full evolution pipeline with detailed debug logging:
1. Ingest initial data → query → verify baseline
2. Ingest conflicting/updating data → query → see confusion
3. Evolve (consolidate, decay, importance) → query → see improvement
4. Verify all evolution stages are working

This demonstrates that GraphMem's evolution makes queries BETTER over time.
"""

import os
import sys
import logging
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Reduce noise from HTTP requests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger("evolution_test")


def print_section(title: str):
    """Print a visible section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_entities(gm, label: str):
    """Print current entities in memory."""
    print(f"\n📊 {label} - Entities ({len(gm._memory.nodes)}):")
    for node in list(gm._memory.nodes.values())[:15]:
        aliases = f" (aliases: {', '.join(list(node.aliases)[:3])})" if node.aliases else ""
        importance = node.importance.name if hasattr(node.importance, 'name') else str(node.importance)
        print(f"   • {node.name} [{node.entity_type}] - importance: {importance}{aliases}")


def print_relationships(gm, label: str):
    """Print current relationships in memory."""
    print(f"\n🔗 {label} - Relationships ({len(gm._memory.edges)}):")
    for edge in list(gm._memory.edges.values())[:15]:
        temporal = ""
        if hasattr(edge, 'valid_from') and edge.valid_from:
            from_str = edge.valid_from.strftime("%Y") if hasattr(edge.valid_from, 'strftime') else str(edge.valid_from)
            until_str = "present" if not edge.valid_until else (edge.valid_until.strftime("%Y") if hasattr(edge.valid_until, 'strftime') else str(edge.valid_until))
            temporal = f" [valid: {from_str} → {until_str}]"
        print(f"   • {edge.source_id[:20]} --[{edge.relation_type}]--> {edge.target_id[:20]}{temporal}")


def print_communities(gm, label: str):
    """Print current communities."""
    clusters = list(gm._memory.clusters.values())
    print(f"\n🏘️ {label} - Communities ({len(clusters)}):")
    for cluster in clusters[:5]:
        cluster_id = str(cluster.id)[:30] if cluster.id else "unknown"
        entities = cluster.entities[:5] if hasattr(cluster, 'entities') and cluster.entities else []
        summary = cluster.summary[:300] if hasattr(cluster, 'summary') and cluster.summary else "No summary"
        print(f"\n   Community: {cluster_id}...")
        print(f"   Entities: {', '.join(entities)}{'...' if len(entities) >= 5 else ''}")
        print(f"   Summary: {summary}...")


def main():
    from graphmem import GraphMem, MemoryConfig
    
    # Configuration - Use Azure OpenAI (set env vars or pass via CLI)
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not api_key or not api_base:
        print("❌ Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables")
        print("   Example:")
        print("   export AZURE_OPENAI_API_KEY='your-key'")
        print("   export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com/'")
        sys.exit(1)
    
    config = MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key=api_key,
        llm_api_base=api_base,
        llm_model="gpt-4.1-mini",
        azure_api_version="2024-08-01-preview",
        azure_deployment="gpt-4.1-mini",
        
        embedding_provider="azure_openai",
        embedding_api_key=api_key,
        embedding_api_base=api_base,
        embedding_model="text-embedding-3-small",
        azure_embedding_deployment="text-embedding-3-small",
        
        turso_db_path="evolution_test.db",
    )
    
    # Clean start
    if os.path.exists("evolution_test.db"):
        os.remove("evolution_test.db")
    if os.path.exists("evolution_test.db_cache.db"):
        os.remove("evolution_test.db_cache.db")
    
    gm = GraphMem(config)
    
    # =========================================================================
    # STAGE 1: Initial Ingestion
    # =========================================================================
    print_section("STAGE 1: INITIAL INGESTION")
    
    # Ingest initial facts about a company with CEO
    initial_docs = [
        {
            "id": "doc1",
            "content": """
            TechCorp Inc. was founded in 2015 by John Smith.
            John Smith served as the CEO of TechCorp from 2015 to 2020.
            Under his leadership, TechCorp grew to 500 employees.
            John Smith, also known as J. Smith or Dr. John Smith, was a pioneer in AI.
            """
        },
        {
            "id": "doc2",
            "content": """
            TechCorp Inc. is headquartered in San Francisco, California.
            The company specializes in artificial intelligence and machine learning.
            TechCorp has raised $100 million in Series B funding in 2019.
            """
        },
    ]
    
    print("📥 Ingesting initial documents...")
    logger.info("=" * 50)
    logger.info("INGESTION STAGE: Extracting entities and relationships")
    logger.info("=" * 50)
    
    for doc in initial_docs:
        gm.ingest(doc["content"])
        print(f"   ✓ Ingested: {doc['id']}")
    
    print_entities(gm, "After Initial Ingestion")
    print_relationships(gm, "After Initial Ingestion")
    
    # =========================================================================
    # STAGE 2: Query BEFORE Evolution
    # =========================================================================
    print_section("STAGE 2: QUERY BEFORE EVOLUTION")
    
    test_queries = [
        "Who is the CEO of TechCorp?",
        "When was TechCorp founded and by whom?",
        "What is Dr. John Smith known for?",  # Uses alias
    ]
    
    print("🔍 Testing queries BEFORE evolution...")
    before_answers = {}
    
    for query in test_queries:
        response = gm.query(query)
        before_answers[query] = response.answer
        print(f"\n   Q: {query}")
        print(f"   A: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Nodes retrieved: {len(response.nodes)}")
    
    # =========================================================================
    # STAGE 3: Ingest CONFLICTING/UPDATED Data
    # =========================================================================
    print_section("STAGE 3: INGEST UPDATED/CONFLICTING DATA")
    
    updated_docs = [
        {
            "id": "doc3",
            "content": """
            BREAKING NEWS (2023): TechCorp Inc. announced leadership change.
            Sarah Johnson is the new CEO of TechCorp since January 2023.
            Sarah Johnson, previously COO, now leads the company as Chief Executive Officer.
            Former CEO John Smith has moved to an advisory board position.
            Sarah Johnson is the CURRENT CEO and will drive TechCorp's AI strategy.
            """
        },
        {
            "id": "doc4",
            "content": """
            TechCorp update: The company now has 2000 employees as of 2023.
            Under CEO Sarah Johnson's leadership, TechCorp expanded to Europe.
            Sarah Johnson is also known as S. Johnson or Dr. Sarah Johnson.
            """
        },
    ]
    
    print("📥 Ingesting updated/conflicting documents...")
    logger.info("=" * 50)
    logger.info("INGESTION STAGE: New conflicting data about CEO")
    logger.info("=" * 50)
    
    for doc in updated_docs:
        gm.ingest(doc["content"])
        print(f"   ✓ Ingested: {doc['id']}")
    
    print_entities(gm, "After Update Ingestion")
    print_relationships(gm, "After Update Ingestion")
    
    # =========================================================================
    # STAGE 4: Query BEFORE Evolution (Expect Confusion)
    # =========================================================================
    print_section("STAGE 4: QUERY AFTER NEW DATA (BEFORE EVOLUTION)")
    
    print("🔍 Testing queries with conflicting data (BEFORE evolution)...")
    print("   ⚠️  Expect some confusion - we have OLD and NEW CEO data!")
    
    for query in test_queries:
        response = gm.query(query)
        print(f"\n   Q: {query}")
        print(f"   A: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
    
    # =========================================================================
    # STAGE 5: EVOLVE - The Magic Happens Here
    # =========================================================================
    print_section("STAGE 5: EVOLVE THE MEMORY")
    
    print("🧠 Running evolution...")
    print("   This will:")
    print("   • Consolidate duplicate entities (John Smith variations)")
    print("   • Apply temporal reasoning (mark old CEO relationships as ended)")
    print("   • Update importance scores (current facts more important)")
    print("   • Generate exhaustive community summaries")
    print()
    
    logger.info("=" * 50)
    logger.info("EVOLUTION STAGE: Consolidation, Decay, Importance")
    logger.info("=" * 50)
    
    evolution_result = gm.evolve()
    
    # evolution_result is an EvolutionResult with .events attribute
    events = evolution_result.events if hasattr(evolution_result, 'events') else evolution_result
    
    print(f"\n📈 Evolution Results:")
    print(f"   • Total events: {len(events)}")
    print(f"   • Consolidation events: {sum(1 for e in events if 'consolidat' in str(e.evolution_type).lower())}")
    print(f"   • Decay events: {sum(1 for e in events if 'decay' in str(e.evolution_type).lower())}")
    
    # Show evolution events
    print(f"\n📋 Evolution Events:")
    for event in events[:10]:
        print(f"   • {event.evolution_type.name}: {event.reason[:60]}...")
    
    print_entities(gm, "After Evolution")
    print_relationships(gm, "After Evolution")
    print_communities(gm, "After Evolution")
    
    # =========================================================================
    # STAGE 6: Query AFTER Evolution (Expect Improvement)
    # =========================================================================
    print_section("STAGE 6: QUERY AFTER EVOLUTION")
    
    print("🔍 Testing queries AFTER evolution...")
    print("   ✅ Expect correct answers about CURRENT state!")
    
    after_answers = {}
    for query in test_queries:
        response = gm.query(query)
        after_answers[query] = response.answer
        print(f"\n   Q: {query}")
        print(f"   A: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Nodes retrieved: {len(response.nodes)}")
    
    # Add more specific queries
    print("\n🔍 Testing specific temporal queries...")
    
    temporal_queries = [
        "Who is the CURRENT CEO of TechCorp?",
        "Who was the CEO of TechCorp in 2018?",
        "When did Sarah Johnson become CEO?",
    ]
    
    for query in temporal_queries:
        response = gm.query(query)
        print(f"\n   Q: {query}")
        print(f"   A: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
    
    # =========================================================================
    # STAGE 7: Summary of Evolution Impact
    # =========================================================================
    print_section("STAGE 7: EVOLUTION IMPACT SUMMARY")
    
    print("📊 BEFORE vs AFTER Evolution:")
    print("-" * 60)
    
    for query in test_queries[:2]:
        print(f"\n   Q: {query}")
        print(f"   BEFORE: {before_answers.get(query, 'N/A')[:100]}...")
        print(f"   AFTER:  {after_answers.get(query, 'N/A')[:100]}...")
    
    print("\n" + "=" * 80)
    print("✅ EVOLUTION TEST COMPLETE")
    print("=" * 80)
    print("""
Key Observations:
1. BEFORE evolution: May return old CEO (John Smith) or be confused
2. AFTER evolution: Should correctly identify Sarah Johnson as CURRENT CEO
3. Temporal queries should respect valid_from/valid_until dates
4. Alias queries (Dr. John Smith) should find consolidated entity
5. Community summaries should explain temporal scope and key facts
    """)


if __name__ == "__main__":
    main()


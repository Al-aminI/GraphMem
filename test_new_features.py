#!/usr/bin/env python3
"""
Quick test of new GraphMem features:
1. Alias extraction
2. Coreference resolution
3. Temporal validity

Run: python test_new_features.py
"""

import os
import sys
sys.path.insert(0, "src")

from graphmem import GraphMem, MemoryConfig

# Use Azure OpenAI - Set these environment variables before running:
# export AZURE_OPENAI_KEY="your-api-key"
# export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
API_KEY = os.getenv("AZURE_OPENAI_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4.1-mini")
EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")

if not API_KEY or not ENDPOINT:
    print("❌ ERROR: Set environment variables first:")
    print("   export AZURE_OPENAI_KEY='your-api-key'")
    print("   export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
    print("   export AZURE_LLM_DEPLOYMENT='gpt-4.1-mini'  # optional")
    print("   export AZURE_EMBED_DEPLOYMENT='text-embedding-3-small'  # optional")
    sys.exit(1)


def test_alias_extraction():
    """Test: Does LLM extract aliases?"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Alias Extraction")
    print("="*60)
    
    # Clean up
    if os.path.exists("test_alias.db"):
        os.remove("test_alias.db")
    
    config = MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key=API_KEY,
        llm_api_base=ENDPOINT,
        llm_model=LLM_DEPLOYMENT,
        embedding_provider="azure_openai",
        embedding_api_key=API_KEY,
        embedding_api_base=ENDPOINT,
        embedding_model=EMBED_DEPLOYMENT,
        azure_api_version="2024-08-01-preview",
        azure_deployment=LLM_DEPLOYMENT,
        azure_embedding_deployment=EMBED_DEPLOYMENT,
        turso_db_path="test_alias.db",
    )
    
    gm = GraphMem(config)
    
    # Document with aliases
    doc = """
    Dr. Alexander Chen, also known as "The Quantum Pioneer", is a renowned researcher.
    Alex Chen (as his friends call him) founded Quantum AI Labs in 2015.
    Professor Chen has published over 100 papers on quantum computing.
    """
    
    print(f"\n📄 Ingesting document with multiple aliases...")
    result = gm.ingest(doc)
    print(f"   Entities: {result.get('entities', 0)}")
    print(f"   Relationships: {result.get('relationships', 0)}")
    
    # Check what was extracted
    print(f"\n🔍 Extracted entities:")
    for node_id, node in gm._memory.nodes.items():
        print(f"   • {node.name} ({node.entity_type})")
        if node.aliases:
            print(f"     Aliases: {node.aliases}")
    
    # Query with different names
    print(f"\n🔍 Testing queries with different names:")
    
    queries = [
        "What did Alexander Chen found?",
        "What did Dr. Chen found?",
        "What did The Quantum Pioneer found?",
        "What did Alex Chen found?",
    ]
    
    for q in queries:
        response = gm.query(q)
        print(f"\n   Q: {q}")
        print(f"   A: {response.answer[:100]}...")
    
    return gm


def test_temporal_validity():
    """Test: Do relationships have valid_from/valid_until?"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Temporal Validity")
    print("="*60)
    
    if os.path.exists("test_temporal.db"):
        os.remove("test_temporal.db")
    
    config = MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key=API_KEY,
        llm_api_base=ENDPOINT,
        llm_model=LLM_DEPLOYMENT,
        embedding_provider="azure_openai",
        embedding_api_key=API_KEY,
        embedding_api_base=ENDPOINT,
        embedding_model=EMBED_DEPLOYMENT,
        azure_api_version="2024-08-01-preview",
        azure_deployment=LLM_DEPLOYMENT,
        azure_embedding_deployment=EMBED_DEPLOYMENT,
        turso_db_path="test_temporal.db",
    )
    
    gm = GraphMem(config)
    
    # Documents with temporal information
    doc1 = """
    NovaTech Industries CEO History:
    - John Mitchell was CEO from 2010 to 2015, when he stepped down.
    - Sarah Williams became CEO in 2015 and served until 2018.
    - David Park was appointed CEO in 2019 and retired in 2021.
    - Emily Zhang took over as CEO in August 2021 until March 2023.
    - Marcus Johnson is the current CEO since March 2023.
    """
    
    print(f"\n📄 Ingesting CEO history...")
    result = gm.ingest(doc1)
    print(f"   Entities: {result.get('entities', 0)}")
    print(f"   Relationships: {result.get('relationships', 0)}")
    
    # Check extracted edges with temporal validity
    print(f"\n🔍 Extracted relationships with temporal validity:")
    for edge_id, edge in gm._memory.edges.items():
        temporal_info = ""
        if edge.valid_from or edge.valid_until:
            temporal_info = f" [valid: {edge.valid_from} → {edge.valid_until}]"
        print(f"   • {edge.source_id[:20]}... → {edge.relation_type} → {edge.target_id[:20]}...{temporal_info}")
    
    # Query for current CEO
    print(f"\n🔍 Testing temporal queries:")
    
    queries = [
        "Who is the current CEO of NovaTech?",
        "Who was the CEO of NovaTech in 2016?",
        "List all CEOs of NovaTech in order.",
    ]
    
    for q in queries:
        response = gm.query(q)
        print(f"\n   Q: {q}")
        print(f"   A: {response.answer[:150]}...")
    
    return gm


def test_coreference_resolution():
    """Test: Does coreference resolution link entities across documents?"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Coreference Resolution")
    print("="*60)
    
    if os.path.exists("test_coref.db"):
        os.remove("test_coref.db")
    
    config = MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key=API_KEY,
        llm_api_base=ENDPOINT,
        llm_model=LLM_DEPLOYMENT,
        embedding_provider="azure_openai",
        embedding_api_key=API_KEY,
        embedding_api_base=ENDPOINT,
        embedding_model=EMBED_DEPLOYMENT,
        azure_api_version="2024-08-01-preview",
        azure_deployment=LLM_DEPLOYMENT,
        azure_embedding_deployment=EMBED_DEPLOYMENT,
        turso_db_path="test_coref.db",
    )
    
    gm = GraphMem(config)
    
    # Document 1: Full name
    doc1 = """
    Alexander Chen is a world-renowned quantum computing researcher.
    He founded Quantum AI Labs in San Francisco in 2015.
    """
    
    print(f"\n📄 Ingesting Document 1 (full name)...")
    result1 = gm.ingest(doc1)
    print(f"   Entities: {result1.get('entities', 0)}")
    
    print(f"\n   Entities after Doc 1:")
    for node_id, node in gm._memory.nodes.items():
        print(f"   • {node.name}")
    
    # Document 2: Uses different names
    doc2 = """
    Dr. Chen recently won the Turing Award for his contributions.
    The Quantum Pioneer, as he's known in the industry, accepted the award in Stockholm.
    He thanked his team at Quantum AI Labs.
    """
    
    print(f"\n📄 Ingesting Document 2 (aliases: Dr. Chen, The Quantum Pioneer)...")
    result2 = gm.ingest(doc2)
    print(f"   Entities: {result2.get('entities', 0)}")
    
    print(f"\n   Entities after Doc 2 (should be consolidated):")
    for node_id, node in gm._memory.nodes.items():
        aliases_str = f" (aliases: {node.aliases})" if node.aliases else ""
        print(f"   • {node.name}{aliases_str}")
    
    # Evolve to consolidate
    print(f"\n🔄 Running evolution to consolidate...")
    gm.evolve()
    
    print(f"\n   Entities after evolution:")
    for node_id, node in gm._memory.nodes.items():
        aliases_str = f" (aliases: {node.aliases})" if node.aliases else ""
        print(f"   • {node.name}{aliases_str}")
    
    # Query with different names
    print(f"\n🔍 Testing cross-document queries:")
    
    queries = [
        "What company did Alexander Chen found?",
        "What award did Dr. Chen win?",
        "What did The Quantum Pioneer win?",
    ]
    
    for q in queries:
        response = gm.query(q)
        print(f"\n   Q: {q}")
        print(f"   A: {response.answer[:100]}...")
    
    return gm


def main():
    print("\n" + "="*60)
    print("🧪 GRAPHMEM NEW FEATURES TEST")
    print("="*60)
    print("\nTesting:")
    print("1. Alias extraction from LLM")
    print("2. Temporal validity on relationships")
    print("3. Coreference resolution across documents")
    
    try:
        test_alias_extraction()
        test_temporal_validity()
        test_coreference_resolution()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print("\nCheck the output above to see if:")
        print("• Aliases are extracted (aliases field populated)")
        print("• Temporal info is captured (valid_from/valid_until)")
        print("• Coreferences are resolved (entities consolidated)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


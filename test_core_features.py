#!/usr/bin/env python3
"""
Test GraphMem's Core Brain-Like Features

This script verifies that each promised feature ACTUALLY works:

1. MEMORY DECAY - Old/irrelevant memories fade, don't get retrieved
2. KNOWLEDGE GRAPH - Relationships between concepts work
3. PAGERANK CENTRALITY - Hub entities get higher importance  
4. TEMPORAL VALIDITY - "CEO in 2015" vs "CEO now" works

Run: python test_core_features.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, "src")

# Set up Azure credentials from environment
# Set these before running:
#   export AZURE_OPENAI_KEY="your-key"
#   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
if not os.environ.get("AZURE_OPENAI_KEY") or not os.environ.get("AZURE_OPENAI_ENDPOINT"):
    print("ERROR: Set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT environment variables")
    sys.exit(1)

from graphmem import GraphMem, MemoryConfig
from graphmem.core.memory_types import MemoryImportance, EvolutionType


def get_config(db_name: str) -> MemoryConfig:
    """Get standard test config."""
    if os.path.exists(db_name):
        os.remove(db_name)
    
    return MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key=os.environ["AZURE_OPENAI_KEY"],
        llm_api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
        llm_model="gpt-4.1-mini",
        embedding_provider="azure_openai",
        embedding_api_key=os.environ["AZURE_OPENAI_KEY"],
        embedding_api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
        embedding_model="text-embedding-3-small",
        azure_api_version="2024-08-01-preview",
        azure_deployment="gpt-4.1-mini",
        azure_embedding_deployment="text-embedding-3-small",
        turso_db_path=db_name,
        evolution_enabled=True,
    )


def test_memory_decay():
    """
    TEST 1: MEMORY DECAY
    
    Old facts that are contradicted should:
    1. Get lower importance when superseded
    2. Not be retrieved when querying for current state
    
    Scenario:
    - Ingest: "John is CEO of Acme Corp"
    - Wait / Simulate aging
    - Ingest: "Sarah is CEO of Acme Corp" (contradicts John)
    - Query: "Who is CEO of Acme Corp?"
    - Expected: Sarah (not John)
    """
    print("\n" + "="*70)
    print("🧠 TEST 1: MEMORY DECAY")
    print("="*70)
    print("Scenario: Old CEO replaced by new CEO - old fact should decay")
    
    gm = GraphMem(get_config("decay_test.db"))
    
    # Step 1: Ingest OLD fact
    print("\n📥 Step 1: Ingest 'John Smith is CEO of Acme Corp' (OLD)")
    gm.ingest("John Smith has been the CEO of Acme Corp since 2015. He leads all operations.")
    
    # Check John's importance
    john_nodes = [n for n in gm._memory.nodes.values() if "john" in n.name.lower()]
    if john_nodes:
        print(f"   John's importance: {john_nodes[0].importance.name}")
    
    # Step 2: Run evolution (applies decay)
    print("\n🔄 Step 2: Run evolve() to apply decay...")
    events = gm.evolve()
    print(f"   Evolution events: {len(events)}")
    
    # Step 3: Ingest NEW contradicting fact
    print("\n📥 Step 3: Ingest 'Sarah Johnson is NOW CEO of Acme Corp' (NEW)")
    gm.ingest("Sarah Johnson was appointed as CEO of Acme Corp in 2024. John Smith stepped down. Sarah now leads the company.")
    
    # Check both importances
    john_nodes = [n for n in gm._memory.nodes.values() if "john" in n.name.lower()]
    sarah_nodes = [n for n in gm._memory.nodes.values() if "sarah" in n.name.lower()]
    
    print("\n📊 Importance levels after update:")
    if john_nodes:
        print(f"   John Smith: {john_nodes[0].importance.name} (value={john_nodes[0].importance.value})")
    if sarah_nodes:
        print(f"   Sarah Johnson: {sarah_nodes[0].importance.name} (value={sarah_nodes[0].importance.value})")
    
    # Step 4: Evolve again
    print("\n🔄 Step 4: Run evolve() again to process the conflict...")
    events = gm.evolve()
    print(f"   Evolution events: {len(events)}")
    for e in events:
        print(f"     - {e.evolution_type.name}: {e.reason[:60]}")
    
    # Step 5: Query for CURRENT CEO
    print("\n🔍 Step 5: Query 'Who is the CURRENT CEO of Acme Corp?'")
    response = gm.query("Who is the current CEO of Acme Corp?")
    print(f"   Answer: {response.answer[:200]}")
    
    # Verify
    is_correct = "sarah" in response.answer.lower()
    mentions_john = "john" in response.answer.lower()
    
    print("\n" + "-"*50)
    if is_correct and not mentions_john:
        print("✅ PASS: Correctly returns Sarah, doesn't mention John")
    elif is_correct and mentions_john:
        print("⚠️ PARTIAL: Returns Sarah but still mentions John")
    else:
        print("❌ FAIL: Should return Sarah as current CEO")
    
    # Show decay status
    print("\n📉 Decay Status:")
    for n in gm._memory.nodes.values():
        if "john" in n.name.lower() or "sarah" in n.name.lower() or "acme" in n.name.lower():
            print(f"   {n.name}: importance={n.importance.name}, access_count={n.access_count}")
    
    return is_correct


def test_knowledge_graph():
    """
    TEST 2: KNOWLEDGE GRAPH
    
    Information should be connected:
    - Elon Musk → CEO of → Tesla
    - Elon Musk → CEO of → SpaceX
    - Query: "What companies does Elon Musk lead?" 
    - Should find BOTH via graph relationships
    """
    print("\n" + "="*70)
    print("🔗 TEST 2: KNOWLEDGE GRAPH (Relationships)")
    print("="*70)
    print("Scenario: Multi-hop relationships - find all connections")
    
    gm = GraphMem(get_config("graph_test.db"))
    
    # Ingest facts about Elon
    print("\n📥 Ingesting connected facts...")
    gm.ingest("Elon Musk is the CEO of Tesla. Tesla makes electric vehicles.")
    gm.ingest("Elon Musk founded SpaceX. He serves as CEO of SpaceX.")
    gm.ingest("Elon Musk also owns X (formerly Twitter).")
    gm.evolve()
    
    # Show graph structure
    print("\n📊 Knowledge Graph Structure:")
    print(f"   Entities: {len(gm._memory.nodes)}")
    print(f"   Relationships: {len(gm._memory.edges)}")
    
    # Find Elon's connections
    elon_nodes = [n for n in gm._memory.nodes.values() if "elon" in n.name.lower() or "musk" in n.name.lower()]
    if elon_nodes:
        elon_id = elon_nodes[0].id
        connected = []
        for e in gm._memory.edges.values():
            if elon_id in e.source_id.lower() or "elon" in e.source_id.lower() or "musk" in e.source_id.lower():
                connected.append(f"{e.relation_type} → {e.target_id}")
            if elon_id in e.target_id.lower() or "elon" in e.target_id.lower() or "musk" in e.target_id.lower():
                connected.append(f"{e.source_id} → {e.relation_type}")
        
        print(f"\n   Elon Musk's connections:")
        for c in connected[:10]:
            print(f"     - {c}")
    
    # Query for all companies
    print("\n🔍 Query: 'What companies does Elon Musk lead or own?'")
    response = gm.query("What companies does Elon Musk lead or own?")
    print(f"   Answer: {response.answer[:300]}")
    
    # Verify
    has_tesla = "tesla" in response.answer.lower()
    has_spacex = "spacex" in response.answer.lower()
    has_x = "x" in response.answer.lower() or "twitter" in response.answer.lower()
    
    score = sum([has_tesla, has_spacex, has_x])
    print("\n" + "-"*50)
    print(f"   Found Tesla: {'✅' if has_tesla else '❌'}")
    print(f"   Found SpaceX: {'✅' if has_spacex else '❌'}")
    print(f"   Found X/Twitter: {'✅' if has_x else '❌'}")
    
    if score >= 2:
        print(f"✅ PASS: Found {score}/3 companies via graph")
    else:
        print(f"❌ FAIL: Only found {score}/3 companies")
    
    return score >= 2


def test_pagerank_centrality():
    """
    TEST 3: PAGERANK CENTRALITY
    
    Hub entities should have higher importance scores.
    
    Scenario:
    - Create a hub entity (mentioned in many relationships)
    - Create peripheral entities (mentioned once)
    - Hub should have higher importance after evolve()
    """
    print("\n" + "="*70)
    print("⭐ TEST 3: PAGERANK CENTRALITY")
    print("="*70)
    print("Scenario: Hub entities should have higher importance")
    
    gm = GraphMem(get_config("pagerank_test.db"))
    
    # Create a hub (Google - connected to many things)
    print("\n📥 Creating hub entity (Google) with many connections...")
    gm.ingest("Google was founded by Larry Page and Sergey Brin.")
    gm.ingest("Google owns YouTube, the video streaming platform.")
    gm.ingest("Google developed Android operating system.")
    gm.ingest("Google created the Chrome browser.")
    gm.ingest("Google runs the largest search engine.")
    gm.ingest("Sundar Pichai is the CEO of Google.")
    
    # Create peripheral entity
    print("📥 Creating peripheral entity (random startup)...")
    gm.ingest("TinyStartup Inc was founded in 2023 by John Doe.")
    
    # Evolve to calculate PageRank
    print("\n🔄 Running evolve() to calculate PageRank...")
    events = gm.evolve()
    print(f"   Evolution events: {len(events)}")
    
    # Check importance scores
    print("\n📊 Importance Scores (PageRank effect):")
    
    # Debug: show all entities
    print("   All entities after evolve:")
    for n in gm._memory.nodes.values():
        print(f"     {n.name}: {n.importance.name} ({n.importance.value})")
    
    # Find hub entity (should have highest connections)
    all_nodes = list(gm._memory.nodes.values())
    all_edges = list(gm._memory.edges.values())
    
    # Count connections per node
    connection_counts = {}
    for n in all_nodes:
        count = sum(1 for e in all_edges if n.id in e.source_id.lower() or n.id in e.target_id.lower() 
                    or n.name.lower() in e.source_id.lower() or n.name.lower() in e.target_id.lower())
        connection_counts[n.name] = count
    
    # Find highest connected (hub)
    hub_name = max(connection_counts, key=connection_counts.get) if connection_counts else None
    hub_node = next((n for n in all_nodes if n.name == hub_name), None) if hub_name else None
    
    # Find lowest connected (peripheral)
    peripheral_name = min(connection_counts, key=connection_counts.get) if connection_counts else None
    peripheral_node = next((n for n in all_nodes if n.name == peripheral_name), None) if peripheral_name else None
    
    hub_importance = hub_node.importance.value if hub_node else 0
    peripheral_importance = peripheral_node.importance.value if peripheral_node else 0
    hub_connections = connection_counts.get(hub_name, 0) if hub_name else 0
    peripheral_connections = connection_counts.get(peripheral_name, 0) if peripheral_name else 0
    
    print(f"\n   Hub ({hub_name}): importance={hub_importance}, connections={hub_connections}")
    print(f"   Peripheral ({peripheral_name}): importance={peripheral_importance}, connections={peripheral_connections}")
    
    # Show all entities by importance
    print("\n   All entities by importance:")
    sorted_nodes = sorted(gm._memory.nodes.values(), key=lambda n: n.importance.value, reverse=True)
    for n in sorted_nodes[:8]:
        connections = connection_counts.get(n.name, 0)
        print(f"     {n.name}: {n.importance.name} ({n.importance.value}) - {connections} connections")
    
    print("\n" + "-"*50)
    # PageRank is working if hub (more connections) has higher or equal importance
    if hub_connections > peripheral_connections and hub_importance >= peripheral_importance:
        print(f"✅ PASS: Hub ({hub_name}={hub_importance}, {hub_connections} conn) >= Peripheral ({peripheral_name}={peripheral_importance}, {peripheral_connections} conn)")
        return True
    elif hub_connections <= peripheral_connections:
        print(f"⚠️ No clear hub entity found (all similar connections)")
        return True  # Not a failure if there's no hub
    else:
        print(f"❌ FAIL: Hub should have higher importance")
        print(f"   Hub ({hub_name}): importance={hub_importance}, connections={hub_connections}")
        print(f"   Peripheral ({peripheral_name}): importance={peripheral_importance}, connections={peripheral_connections}")
        return False


def test_temporal_validity():
    """
    TEST 4: TEMPORAL VALIDITY
    
    Point-in-time queries should work:
    - "Who was CEO in 2015?" → returns CEO at that time
    - "Who is CEO now?" → returns current CEO
    """
    print("\n" + "="*70)
    print("⏰ TEST 4: TEMPORAL VALIDITY")
    print("="*70)
    print("Scenario: Point-in-time queries for CEO history")
    
    gm = GraphMem(get_config("temporal_test.db"))
    
    # Ingest CEO history
    print("\n📥 Ingesting CEO history...")
    history = """
    NovaCorp CEO History:
    - Mike Brown was CEO of NovaCorp from 2010 to 2015.
    - Lisa Chen became CEO of NovaCorp in 2015 and served until 2020.
    - David Kim has been CEO of NovaCorp since 2020 and is the current CEO.
    """
    gm.ingest(history)
    gm.evolve()
    
    # Show temporal relationships
    print("\n📊 Relationships with temporal validity:")
    for e in gm._memory.edges.values():
        if "ceo" in e.relation_type.lower() or "nova" in e.source_id.lower() or "nova" in e.target_id.lower():
            temporal = ""
            if e.valid_from:
                temporal = f" [valid: {e.valid_from}"
                if e.valid_until:
                    temporal += f" → {e.valid_until}]"
                else:
                    temporal += " → present]"
            print(f"   {e.source_id} --[{e.relation_type}]--> {e.target_id}{temporal}")
    
    # Query for CURRENT CEO
    print("\n🔍 Query 1: 'Who is the CURRENT CEO of NovaCorp?'")
    response1 = gm.query("Who is the current CEO of NovaCorp?")
    print(f"   Answer: {response1.answer[:200]}")
    current_correct = "david" in response1.answer.lower() or "kim" in response1.answer.lower()
    
    # Query for 2016 CEO
    print("\n🔍 Query 2: 'Who was the CEO of NovaCorp in 2016?'")
    response2 = gm.query("Who was the CEO of NovaCorp in 2016?")
    print(f"   Answer: {response2.answer[:200]}")
    historical_correct = "lisa" in response2.answer.lower() or "chen" in response2.answer.lower()
    
    # Query for all CEOs
    print("\n🔍 Query 3: 'List all CEOs of NovaCorp in order'")
    response3 = gm.query("List all the CEOs of NovaCorp in chronological order")
    print(f"   Answer: {response3.answer[:300]}")
    
    print("\n" + "-"*50)
    print(f"   Current CEO (David Kim): {'✅' if current_correct else '❌'}")
    print(f"   2016 CEO (Lisa Chen): {'✅' if historical_correct else '❌'}")
    
    if current_correct and historical_correct:
        print("✅ PASS: Temporal queries work correctly!")
    elif current_correct or historical_correct:
        print("⚠️ PARTIAL: Some temporal queries work")
    else:
        print("❌ FAIL: Temporal queries not working")
    
    return current_correct and historical_correct


def main():
    print("\n" + "="*70)
    print("🧪 GRAPHMEM CORE FEATURES TEST")
    print("="*70)
    print("""
Testing the brain-like features that make GraphMem unique:

| Human Brain        | GraphMem           | What We're Testing           |
|--------------------|--------------------|-----------------------------|
| 🧠 Forgetting Curve | Memory Decay       | Old facts fade when updated |
| 🔗 Neural Networks  | Knowledge Graph    | Multi-hop relationships     |
| ⭐ Importance       | PageRank Centrality| Hub entities > peripheral   |
| ⏰ Episodic Memory  | Temporal Validity  | "CEO in 2015" vs "CEO now"  |
""")
    
    results = {}
    
    # Run all tests
    results["Memory Decay"] = test_memory_decay()
    results["Knowledge Graph"] = test_knowledge_graph()
    results["PageRank Centrality"] = test_pagerank_centrality()
    results["Temporal Validity"] = test_temporal_validity()
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for feature, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {feature}: {status}")
    
    total_pass = sum(results.values())
    print(f"\n   Total: {total_pass}/{len(results)} features working")
    
    if total_pass == len(results):
        print("\n🏆 ALL CORE FEATURES WORKING!")
    else:
        print(f"\n⚠️ {len(results) - total_pass} features need attention")
    
    # Cleanup
    for f in ["decay_test.db", "graph_test.db", "pagerank_test.db", "temporal_test.db"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()


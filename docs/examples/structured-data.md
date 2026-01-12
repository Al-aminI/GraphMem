# Structured Data Ingestion

Learn how to ingest custom entities and relationships directly into GraphMem without relying on LLM extraction.

!!! info "Use Cases"
    - You already have structured data (databases, APIs, CRMs)
    - You want to define custom entity types (Person, Organization, Quote)
    - You have known relationships ("person works at organization")
    - You need precise control over the knowledge graph

---

## Quick Start: Structured Ingestion

```python
from graphmem import GraphMem, MemoryConfig
from graphmem.core.memory_types import MemoryNode, MemoryEdge, MemoryImportance
from datetime import datetime

# Initialize
config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    turso_db_path="my_structured_memory.db",
)

memory = GraphMem(config, memory_id="my_crm", user_id="default")
memory._ensure_initialized()  # Initialize internal state

# Create custom entities
person = MemoryNode(
    id="person_john_doe",
    name="John Doe",
    entity_type="Person",
    description="Senior Software Engineer at Acme Corp, 5 years experience",
    aliases={"John", "J. Doe", "Johnny"},
    properties={
        "email": "john.doe@acme.com",
        "department": "Engineering",
        "hire_date": "2019-03-15",
    },
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)

organization = MemoryNode(
    id="org_acme_corp",
    name="Acme Corporation",
    entity_type="Organization",
    description="Technology company specializing in cloud infrastructure",
    aliases={"Acme Corp", "Acme", "ACME"},
    properties={
        "industry": "Technology",
        "founded": "2010",
        "headquarters": "San Francisco, CA",
    },
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)

# Add entities to memory
memory._memory.add_node(person)
memory._memory.add_node(organization)

# Create relationships
works_at = MemoryEdge(
    id="rel_john_works_at_acme",
    source_id=person.id,
    target_id=organization.id,
    relation_type="works_at",
    description="John Doe is employed at Acme Corporation as Senior Software Engineer",
    properties={
        "role": "Senior Software Engineer",
        "department": "Engineering",
    },
    valid_from=datetime(2019, 3, 15),  # When relationship started
    valid_until=None,  # Still active (None = current)
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
)

memory._memory.add_edge(works_at)

# Save to persistent storage
memory._save_memory()

# Now query normally!
response = memory.query("Who works at Acme Corporation?")
print(response.answer)  # "John Doe works at Acme Corporation as Senior Software Engineer"
```

---

## Custom Entity Types

### Person

```python
person = MemoryNode(
    id="person_jane_smith",
    name="Jane Smith",
    entity_type="Person",
    description="CEO of TechStart Inc., former VP at Google",
    aliases={"Jane", "J. Smith", "Dr. Jane Smith"},
    properties={
        "title": "CEO",
        "email": "jane@techstart.io",
        "linkedin": "linkedin.com/in/janesmith",
        "expertise": ["AI", "Product Management", "Strategy"],
    },
    importance=MemoryImportance.CRITICAL,  # Key person
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
```

### Organization

```python
org = MemoryNode(
    id="org_techstart",
    name="TechStart Inc.",
    entity_type="Organization",
    description="AI startup focused on enterprise automation",
    aliases={"TechStart", "Tech Start", "TSI"},
    properties={
        "industry": "Artificial Intelligence",
        "founded": "2022",
        "funding": "$50M Series B",
        "employees": 150,
        "headquarters": "Austin, TX",
    },
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
```

### Quote

```python
quote = MemoryNode(
    id="quote_vision_statement",
    name="AI will transform every industry",
    entity_type="Quote",
    description="Vision statement from CEO Jane Smith at TechCrunch 2024",
    properties={
        "speaker": "Jane Smith",
        "context": "TechCrunch Disrupt 2024 Keynote",
        "date": "2024-10-15",
        "full_text": "AI will transform every industry, and we're just at the beginning.",
    },
    importance=MemoryImportance.MEDIUM,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
```

### Product

```python
product = MemoryNode(
    id="product_automate_pro",
    name="AutomatePro",
    entity_type="Product",
    description="Enterprise workflow automation platform powered by AI",
    aliases={"Automate Pro", "AP", "AutoPro"},
    properties={
        "version": "3.2",
        "launch_date": "2024-06-01",
        "pricing": "Enterprise",
        "features": ["Workflow Builder", "AI Assistant", "Analytics"],
    },
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
```

### Event

```python
event = MemoryNode(
    id="event_series_b",
    name="TechStart Series B Funding",
    entity_type="Event",
    description="$50M Series B funding round led by Sequoia Capital",
    properties={
        "date": "2024-03-01",
        "amount": "$50M",
        "lead_investor": "Sequoia Capital",
        "participants": ["Andreessen Horowitz", "Y Combinator"],
    },
    importance=MemoryImportance.HIGH,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
```

---

## Relationship Types

### Employment Relationships

```python
# Current employment
works_at = MemoryEdge(
    id="rel_john_works_at_acme",
    source_id="person_john_doe",
    target_id="org_acme_corp",
    relation_type="works_at",
    description="Current employment",
    properties={"role": "Senior Engineer", "department": "Engineering"},
    valid_from=datetime(2019, 3, 15),
    valid_until=None,  # Still employed
    memory_id=memory.memory_id,
)

# Past employment (with end date)
worked_at = MemoryEdge(
    id="rel_jane_worked_at_google",
    source_id="person_jane_smith",
    target_id="org_google",
    relation_type="worked_at",
    description="Former VP of Product at Google",
    properties={"role": "VP of Product"},
    valid_from=datetime(2015, 1, 1),
    valid_until=datetime(2022, 6, 30),  # Left in 2022
    memory_id=memory.memory_id,
)
```

### Leadership Relationships

```python
# CEO relationship
leads = MemoryEdge(
    id="rel_jane_leads_techstart",
    source_id="person_jane_smith",
    target_id="org_techstart",
    relation_type="leads",
    description="Jane Smith is CEO of TechStart",
    properties={"role": "CEO", "board_member": True},
    valid_from=datetime(2022, 7, 1),
    valid_until=None,  # Current
    memory_id=memory.memory_id,
)

# Reports to relationship
reports_to = MemoryEdge(
    id="rel_john_reports_to_jane",
    source_id="person_john_doe",
    target_id="person_jane_smith",
    relation_type="reports_to",
    description="Direct reporting relationship",
    valid_from=datetime(2023, 1, 1),
    memory_id=memory.memory_id,
)
```

### Ownership/Investment Relationships

```python
# Investment relationship
invested_in = MemoryEdge(
    id="rel_sequoia_invested_techstart",
    source_id="org_sequoia",
    target_id="org_techstart",
    relation_type="invested_in",
    description="Led Series B round",
    properties={"round": "Series B", "amount": "$50M", "lead": True},
    valid_from=datetime(2024, 3, 1),
    memory_id=memory.memory_id,
)

# Product ownership
develops = MemoryEdge(
    id="rel_techstart_develops_automatepro",
    source_id="org_techstart",
    target_id="product_automate_pro",
    relation_type="develops",
    description="TechStart develops AutomatePro",
    memory_id=memory.memory_id,
)
```

### Quote/Statement Relationships

```python
# Person said quote
said = MemoryEdge(
    id="rel_jane_said_quote",
    source_id="person_jane_smith",
    target_id="quote_vision_statement",
    relation_type="said",
    description="Statement made at TechCrunch 2024",
    properties={"context": "Keynote speech"},
    valid_from=datetime(2024, 10, 15),
    memory_id=memory.memory_id,
)
```

---

## Batch Import from Structured Sources

### From a Database/CRM

```python
import json

# Example: Import from a JSON export
crm_data = {
    "contacts": [
        {"id": "1", "name": "John Doe", "company": "Acme Corp", "role": "Engineer"},
        {"id": "2", "name": "Jane Smith", "company": "TechStart", "role": "CEO"},
    ],
    "companies": [
        {"id": "c1", "name": "Acme Corp", "industry": "Technology"},
        {"id": "c2", "name": "TechStart", "industry": "AI"},
    ],
}

def import_crm_data(memory: GraphMem, crm_data: dict):
    """Import CRM data as structured entities and relationships."""
    
    memory._ensure_initialized()
    
    # Create company entities
    company_ids = {}
    for company in crm_data["companies"]:
        node = MemoryNode(
            id=f"org_{company['id']}",
            name=company["name"],
            entity_type="Organization",
            description=f"{company['industry']} company",
            properties={"industry": company["industry"]},
            importance=MemoryImportance.HIGH,
            memory_id=memory.memory_id,
            user_id=memory.user_id,
        )
        memory._memory.add_node(node)
        company_ids[company["name"]] = node.id
    
    # Create person entities and relationships
    for contact in crm_data["contacts"]:
        # Create person
        person = MemoryNode(
            id=f"person_{contact['id']}",
            name=contact["name"],
            entity_type="Person",
            description=f"{contact['role']} at {contact['company']}",
            properties={"role": contact["role"]},
            importance=MemoryImportance.MEDIUM,
            memory_id=memory.memory_id,
            user_id=memory.user_id,
        )
        memory._memory.add_node(person)
        
        # Create works_at relationship
        if contact["company"] in company_ids:
            edge = MemoryEdge(
                id=f"rel_{contact['id']}_works_at_{contact['company']}",
                source_id=person.id,
                target_id=company_ids[contact["company"]],
                relation_type="works_at",
                description=f"{contact['name']} works at {contact['company']}",
                properties={"role": contact["role"]},
                importance=MemoryImportance.MEDIUM,
                memory_id=memory.memory_id,
            )
            memory._memory.add_edge(edge)
    
    # Save all to storage
    memory._save_memory()
    
    print(f"Imported {len(crm_data['companies'])} companies, {len(crm_data['contacts'])} contacts")

# Use it
import_crm_data(memory, crm_data)
```

### From a CSV File

```python
import csv

def import_from_csv(memory: GraphMem, csv_path: str, entity_type: str):
    """Import entities from a CSV file."""
    
    memory._ensure_initialized()
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = MemoryNode(
                id=f"{entity_type.lower()}_{row['id']}",
                name=row["name"],
                entity_type=entity_type,
                description=row.get("description", ""),
                properties={k: v for k, v in row.items() if k not in ["id", "name", "description"]},
                importance=MemoryImportance.MEDIUM,
                memory_id=memory.memory_id,
                user_id=memory.user_id,
            )
            memory._memory.add_node(node)
    
    memory._save_memory()
```

---

## Hybrid Approach: Structured + LLM Extraction

Combine structured data with natural language for rich context:

```python
# 1. First, add known structured entities
memory._ensure_initialized()

# Add known entities
ceo = MemoryNode(
    id="person_ceo",
    name="Sarah Chen",
    entity_type="Person",
    description="CEO of Quantum Innovations",
    aliases={"Sarah", "Dr. Chen", "S. Chen"},
    properties={"title": "CEO", "linkedin": "linkedin.com/in/sarahchen"},
    importance=MemoryImportance.CRITICAL,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
memory._memory.add_node(ceo)

company = MemoryNode(
    id="org_quantum",
    name="Quantum Innovations",
    entity_type="Organization",
    description="Quantum computing startup",
    aliases={"Quantum", "QI"},
    importance=MemoryImportance.CRITICAL,
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)
memory._memory.add_node(company)

memory._save_memory()

# 2. Then ingest natural language that references these entities
# GraphMem will automatically link to existing entities via alias matching!
memory.ingest("""
    In today's earnings call, Sarah Chen announced Quantum Innovations achieved 
    $100M ARR, a 150% increase from last year. The company plans to expand 
    into Europe by Q2 2025.
""")

# The LLM extraction will recognize "Sarah Chen" and "Quantum Innovations"
# and link them to your pre-existing structured entities!
```

---

## Generating Embeddings for Custom Entities

For semantic search to work, entities need embeddings:

```python
def add_entity_with_embedding(memory: GraphMem, node: MemoryNode):
    """Add entity and generate its embedding."""
    
    memory._ensure_initialized()
    
    # Generate embedding from name + description
    text_to_embed = f"{node.name}: {node.description}"
    
    embedding = memory._embedding_provider.embed(text_to_embed)
    node.embedding = embedding
    
    memory._memory.add_node(node)
    memory._save_memory()

# Usage
person = MemoryNode(
    id="person_new",
    name="Alex Johnson",
    entity_type="Person",
    description="Product Manager at CloudTech, specializes in AI products",
    memory_id=memory.memory_id,
    user_id=memory.user_id,
)

add_entity_with_embedding(memory, person)

# Now semantic search will find this entity!
response = memory.query("Who works on AI products?")
```

---

## Best Practices

### 1. Use Consistent ID Patterns

```python
# ✅ Good: Consistent, readable IDs
"person_john_doe"
"org_acme_corp"
"product_automate_pro"
"event_series_b_2024"

# ❌ Bad: Random or inconsistent IDs
"12345"
"John"
"ORG_ACME"
```

### 2. Add Rich Aliases

```python
# ✅ Good: Many aliases for better matching
aliases={"John Doe", "John", "J. Doe", "Johnny", "JD"}

# ❌ Bad: Only the canonical name
aliases={"John Doe"}
```

### 3. Use Temporal Validity

```python
# ✅ Good: Track when relationships are valid
valid_from=datetime(2020, 1, 1),
valid_until=datetime(2023, 12, 31),  # Past relationship

valid_from=datetime(2024, 1, 1),
valid_until=None,  # Current relationship (None = present)

# ❌ Bad: No temporal information
# (You lose the ability to query "Who was CEO in 2020?")
```

### 4. Set Appropriate Importance

```python
# ✅ Good: Key entities get high importance
MemoryImportance.CRITICAL  # CEOs, key customers, critical data
MemoryImportance.HIGH      # Important entities
MemoryImportance.MEDIUM    # Regular entities (default)
MemoryImportance.LOW       # Less important, may decay
MemoryImportance.EPHEMERAL # Temporary, will decay quickly
```

### 5. Always Call `_save_memory()` After Bulk Operations

```python
# ✅ Good: Single save after all additions
for entity in entities:
    memory._memory.add_node(entity)
for relation in relations:
    memory._memory.add_edge(relation)
memory._save_memory()  # One save at the end

# ❌ Bad: Save after each addition (slow)
for entity in entities:
    memory._memory.add_node(entity)
    memory._save_memory()  # Slow!
```

---

## Coming Soon: Public Structured API

We're working on a cleaner public API for structured data:

```python
# 🔜 Future API (not yet available)
memory.add_entity(
    name="John Doe",
    entity_type="Person",
    properties={"role": "Engineer"},
    relationships=[
        {"target": "Acme Corp", "type": "works_at", "since": "2020-01-01"}
    ]
)

memory.add_relationship(
    source="John Doe",
    target="Acme Corp",
    relationship="works_at",
    valid_from="2020-01-01",
)
```

Want this feature? [Vote on GitHub Issues](https://github.com/Al-aminI/GraphMem/issues)!

---

## Summary

| Approach | Best For |
|----------|----------|
| **`memory.ingest(text)`** | Natural language, documents, unstructured content |
| **`memory._memory.add_node()`** | Pre-existing structured data, custom entity types |
| **Hybrid** | Structured base + natural language enrichment |

GraphMem supports both automatic LLM extraction AND manual structured data ingestion. Use whichever fits your use case, or combine them for the best of both worlds!

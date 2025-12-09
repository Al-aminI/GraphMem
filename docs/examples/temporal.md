# Temporal Query Examples

Query data at specific points in time.

## CEO Transitions

```python
from graphmem import GraphMem, MemoryConfig

config = MemoryConfig(
    llm_provider="openai",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    embedding_provider="openai",
    embedding_api_key="sk-...",
    embedding_model="text-embedding-3-small",
    turso_db_path="company.db",
)

memory = GraphMem(config, memory_id="temporal_example", user_id="default")

# Ingest historical data
memory.ingest("""
    John Smith was CEO of ACME Corp from January 2010 to June 2018.
    He led the company through its IPO in 2015.
""")

memory.ingest("""
    Jane Doe became CEO of ACME Corp in July 2018.
    She is currently leading the company's AI transformation.
""")

# Query by time period
response = memory.query("Who was CEO in 2015?")
print(response.answer)  # "John Smith"

response = memory.query("Who was CEO in 2019?")
print(response.answer)  # "Jane Doe"

response = memory.query("Who is the current CEO?")
print(response.answer)  # "Jane Doe"
```

## Policy Changes

```python
# Initial policy
memory.ingest("""
    RETURN POLICY (Effective January 2020)
    - Returns accepted within 30 days of purchase
    - Refunds processed within 5-7 business days
    - No restocking fee for unopened items
""")

# Policy update
memory.ingest("""
    UPDATED RETURN POLICY (Effective June 2024)
    - Returns accepted within 60 days of purchase
    - Refunds processed within 2-3 business days
    - No restocking fee for any condition
""")

# Evolve to resolve conflicts
memory.evolve()

# Queries return current policy
response = memory.query("What is the return window?")
print(response.answer)  # "60 days"

# Can still ask about historical policy
response = memory.query("What was the return policy in 2022?")
print(response.answer)  # "30 days"
```

## Employment History

```python
memory.ingest("""
    Alice Johnson's Employment History:
    - Google (2015-2018): Software Engineer
    - Meta (2018-2021): Senior Software Engineer
    - Anthropic (2021-present): Staff Engineer
""")

response = memory.query("Where did Alice work in 2017?")
print(response.answer)  # "Google"

response = memory.query("Where did Alice work in 2020?")
print(response.answer)  # "Meta"

response = memory.query("Where does Alice work now?")
print(response.answer)  # "Anthropic"
```

## Product Pricing

```python
memory.ingest("""
    Product pricing history for Widget Pro:
    - 2020: $49.99
    - 2021: $59.99 (price increase due to chip shortage)
    - 2022: $54.99 (price reduction)
    - 2023: $64.99 (new features added)
    - 2024: $69.99 (current price)
""")

response = memory.query("What was the price in 2021?")
print(response.answer)  # "$59.99"

response = memory.query("What is the current price?")
print(response.answer)  # "$69.99"

response = memory.query("When did the price increase to $64.99?")
print(response.answer)  # "2023"
```

## Contract Management

```python
memory.ingest("""
    Contract #1234 with Vendor ABC:
    - Effective: January 1, 2023
    - Expires: December 31, 2025
    - Annual value: $1.2M
    - Auto-renewal: Yes
""")

memory.ingest("""
    Contract #5678 with Vendor XYZ:
    - Effective: July 1, 2022
    - Expired: June 30, 2024
    - Status: Not renewed
""")

response = memory.query("Is contract 1234 still active?")
print(response.answer)  # "Yes, expires December 31, 2025"

response = memory.query("Is contract 5678 active?")
print(response.answer)  # "No, expired June 30, 2024"

response = memory.query("What contracts were active in 2023?")
print(response.answer)  # "Contract #1234 and #5678"
```

## Direct Edge API

For more control, use the edge API directly:

```python
from graphmem.core.memory_types import MemoryEdge
from datetime import datetime

# Create edge with temporal bounds
edge = MemoryEdge(
    id="john_ceo_acme",
    source_id="john_smith",
    target_id="acme_corp",
    relation_type="CEO_OF",
    valid_from=datetime(2010, 1, 1),
    valid_until=datetime(2018, 6, 30),
)

# Check validity at a point in time
edge.is_valid_at(datetime(2015, 6, 1))  # True
edge.is_valid_at(datetime(2019, 1, 1))  # False

# Mark relationship as ended
edge.supersede(end_time=datetime(2018, 6, 30))
```


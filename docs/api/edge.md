# MemoryEdge

Represents a relationship in the knowledge graph.

## Definition

```python
@dataclass
class MemoryEdge:
    id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    valid_from: datetime = None
    valid_until: datetime = None
    user_id: str = "default"
    memory_id: str = "default"
    properties: dict = None
    priority: int = 0
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `source_id` | str | Source entity ID |
| `target_id` | str | Target entity ID |
| `relation_type` | str | Relationship type |
| `description` | str | Relationship description |
| `weight` | float | Relationship strength |
| `valid_from` | datetime | When relationship started |
| `valid_until` | datetime | When relationship ended (None = current) |
| `user_id` | str | Owner user ID |
| `memory_id` | str | Memory context ID |
| `properties` | dict | Additional properties |
| `priority` | int | Priority for conflict resolution |

## Example

```python
from graphmem.core.memory_types import MemoryEdge
from datetime import datetime

edge = MemoryEdge(
    id="elon_ceo_tesla",
    source_id="elon_musk_001",
    target_id="tesla_001",
    relation_type="CEO_OF",
    description="Elon Musk is CEO of Tesla",
    valid_from=datetime(2008, 10, 1),
    valid_until=None,  # Still current
    user_id="alice",
    memory_id="chat",
)
```

## Methods

### is_valid_at()

Check if relationship is valid at a specific time.

```python
is_valid = edge.is_valid_at(datetime(2015, 6, 1))
```

**Parameters:**
- `query_time` (datetime): Time to check

**Returns:** `bool`

### supersede()

Mark relationship as ended.

```python
edge.supersede(end_time=datetime(2025, 1, 1))
```

**Parameters:**
- `end_time` (datetime): When the relationship ended

### to_dict()

Convert to dictionary.

```python
data = edge.to_dict()
```

### from_dict()

Create from dictionary.

```python
edge = MemoryEdge.from_dict(data)
```

## Temporal Validity

Edges support temporal validity for point-in-time queries:

```python
# CEO from 2010 to 2018
john_ceo = MemoryEdge(
    id="john_ceo",
    source_id="john_smith",
    target_id="acme_corp",
    relation_type="CEO_OF",
    valid_from=datetime(2010, 1, 1),
    valid_until=datetime(2018, 6, 30),
)

# Current CEO
jane_ceo = MemoryEdge(
    id="jane_ceo",
    source_id="jane_doe",
    target_id="acme_corp",
    relation_type="CEO_OF",
    valid_from=datetime(2018, 7, 1),
    valid_until=None,  # Still current
)

# Query
john_ceo.is_valid_at(datetime(2015, 1, 1))  # True
jane_ceo.is_valid_at(datetime(2015, 1, 1))  # False
jane_ceo.is_valid_at(datetime.utcnow())     # True
```


# MemoryNode

Represents an entity in the knowledge graph.

## Definition

```python
@dataclass
class MemoryNode:
    id: str
    name: str
    entity_type: str
    description: str = ""
    embedding: list[float] = None
    importance: float = 1.0
    access_count: int = 0
    last_accessed: datetime = None
    created_at: datetime = None
    user_id: str = "default"
    memory_id: str = "default"
    aliases: list[str] = None
    properties: dict = None
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `name` | str | Entity name |
| `entity_type` | str | Type (Person, Company, etc.) |
| `description` | str | Entity description |
| `embedding` | list[float] | Vector embedding |
| `importance` | float | Importance score (0-1) |
| `access_count` | int | Number of times accessed |
| `last_accessed` | datetime | Last access time |
| `created_at` | datetime | Creation time |
| `user_id` | str | Owner user ID |
| `memory_id` | str | Memory context ID |
| `aliases` | list[str] | Alternative names |
| `properties` | dict | Additional properties |

## Example

```python
from graphmem.core.memory_types import MemoryNode
from datetime import datetime

node = MemoryNode(
    id="elon_musk_001",
    name="Elon Musk",
    entity_type="Person",
    description="CEO of Tesla and SpaceX",
    importance=1.0,
    aliases=["Musk", "Elon"],
    properties={
        "title": "CEO",
        "companies": ["Tesla", "SpaceX", "Neuralink"],
    },
    user_id="alice",
    memory_id="chat",
)
```

## Methods

### to_dict()

Convert to dictionary.

```python
data = node.to_dict()
```

### from_dict()

Create from dictionary.

```python
node = MemoryNode.from_dict(data)
```


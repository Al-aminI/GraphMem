# Installation

## Requirements

- Python 3.9 or higher
- An LLM API key (OpenAI, Azure, Anthropic, or any OpenAI-compatible provider)

## Quick Install

=== "Core (Development)"

    ```bash
    pip install agentic-graph-mem
    ```
    
    !!! note "In-Memory Only"
        The core package uses in-memory storage. Data is lost when your program exits.

=== "With Turso (Recommended)"

    ```bash
    pip install "agentic-graph-mem[libsql]"
    ```
    
    !!! success "Best for Most Users"
        Turso provides SQLite-based persistence that:
        
        - Works offline
        - Persists data to a local file
        - Supports native vector search
        - Requires no server setup

=== "Enterprise (Full Stack)"

    ```bash
    pip install "agentic-graph-mem[all]"
    ```
    
    !!! tip "For Production"
        Includes Neo4j and Redis for:
        
        - Complex graph queries
        - High-performance caching
        - Horizontal scaling
        - ACID transactions

## Package Options

| Package | Command | Use Case |
|---------|---------|----------|
| Core | `pip install agentic-graph-mem` | Development, testing |
| Turso | `pip install "agentic-graph-mem[libsql]"` | Edge, offline, simple apps |
| Neo4j | `pip install "agentic-graph-mem[neo4j]"` | Graph-heavy workloads |
| Redis | `pip install "agentic-graph-mem[redis]"` | High-performance caching |
| All | `pip install "agentic-graph-mem[all]"` | Full enterprise stack |

## Verify Installation

```python
from graphmem import GraphMem, MemoryConfig

# Check version
import graphmem
print(f"GraphMem version: {graphmem.__version__}")
```

## Next Steps

- [Quick Start](quickstart.md) - Get up and running in 5 minutes
- [Configuration](configuration.md) - Configure GraphMem for your needs
- [Storage Backends](storage.md) - Choose the right storage for your use case


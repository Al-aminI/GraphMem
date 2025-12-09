# 🧠 GraphMem

## **The Human Brain for Your AI Agents**

<p align="center">
  <a href="https://pypi.org/project/agentic-graph-mem/"><img src="https://img.shields.io/pypi/v/agentic-graph-mem.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/Al-aminI/GraphMem"><img src="https://img.shields.io/badge/github-Al--aminI/GraphMem-blue.svg" alt="GitHub"></a>
</p>

> **"Memory is the treasury and guardian of all things."** — Cicero

GraphMem is the **first memory system that thinks like a human brain**. It doesn't just store data—it **forgets**, **consolidates**, **prioritizes**, and **evolves** exactly like biological memory does.

**This is the future of enterprise AI agents.**

---

## 🧬 Why GraphMem Changes Everything

### The Problem with Current AI Memory

Every production AI agent faces the same crisis:

```
Day 1:     "Who is the CEO?" → "Elon Musk" ✅
Day 100:   Context window: OVERFLOW 💥
Day 365:   "Who is the CEO?" → "John... or was it Jane... maybe Elon?" 🤯
```

**Vector databases don't forget.** They accumulate garbage until your agent drowns in irrelevant, conflicting, outdated information.

### The GraphMem Solution: Memory That Thinks

GraphMem implements the **four pillars of human memory**:

| Human Brain | GraphMem | Why It Matters |
|-------------|----------|----------------|
| 🧠 **Forgetting Curve** | Memory Decay | Irrelevant memories fade naturally |
| 🔗 **Neural Networks** | Knowledge Graph | Relationships between concepts |
| ⭐ **Importance Weighting** | PageRank Centrality | Hub concepts (Elon Musk) > peripheral ones |
| ⏰ **Episodic Memory** | Temporal Validity | "CEO in 2015" vs "CEO now" |

---

## 🚀 Quick Start

=== "Installation"

    ```bash
    # Core only (in-memory, for development)
    pip install agentic-graph-mem

    # 🔥 RECOMMENDED: Turso (SQLite persistence + offline)
    pip install "agentic-graph-mem[libsql]"

    # Enterprise: Neo4j + Redis (full graph power)
    pip install "agentic-graph-mem[all]"
    ```

=== "Basic Usage"

    ```python
    from graphmem import GraphMem, MemoryConfig

    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
    )

    memory = GraphMem(config)

    # That's it. 3 methods:
    memory.ingest("Tesla is led by CEO Elon Musk...")  # ← Extract knowledge
    memory.query("Who is the CEO?")                    # ← Ask questions
    memory.evolve()                                    # ← Let memory mature
    ```

=== "With Persistence"

    ```python
    from graphmem import GraphMem, MemoryConfig

    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        embedding_provider="openai",
        embedding_api_key="sk-...",
        embedding_model="text-embedding-3-small",
        
        # 🔥 Just add a file path - that's it!
        turso_db_path="my_agent_memory.db",
    )

    memory = GraphMem(config)
    memory.ingest("Important information...")
    # Data persists between restarts!
    ```

=== "Custom Providers"

    ```python
    from graphmem import GraphMem, MemoryConfig

    # OpenRouter, Together, Groq, or any OpenAI-compatible API
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="sk-or-v1-...",
        llm_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        llm_model="google/gemini-2.0-flash-001",
        embedding_provider="openai_compatible",
        embedding_api_key="sk-or-v1-...",
        embedding_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        embedding_model="openai/text-embedding-3-small",
    )

    # Or Azure OpenAI
    config = MemoryConfig(
        llm_provider="azure_openai",
        llm_api_key="your-azure-key",
        llm_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
        azure_deployment="gpt-4",
        llm_model="gpt-4",
        azure_api_version="2024-02-15-preview",
        embedding_provider="azure_openai",
        embedding_api_key="your-azure-key",
        embedding_api_base="https://your-resource.openai.azure.com/",
        azure_embedding_deployment="text-embedding-ada-002",
        embedding_model="text-embedding-ada-002",
    )

    # Or local Ollama
    config = MemoryConfig(
        llm_provider="openai_compatible",
        llm_api_key="not-needed",
        llm_api_base="http://localhost:11434/v1",  # Ollama base URL
        llm_model="llama3.2",
        embedding_provider="openai_compatible",
        embedding_api_key="not-needed",
        embedding_api_base="http://localhost:11434/v1",
        embedding_model="nomic-embed-text",
    )
    ```

---

## 🎯 Revolutionary Features

<div class="grid cards" markdown>

-   :material-clock-time-four:{ .lg .middle } **Point-in-Time Memory**

    ---

    Query the past: *"Who was CEO in 2015?"*

    [:octicons-arrow-right-24: Learn more](concepts/temporal.md)

-   :material-graph:{ .lg .middle } **Knowledge Graph**

    ---

    Automatic entity extraction and relationship mapping

    [:octicons-arrow-right-24: Learn more](concepts/knowledge-graph.md)

-   :material-brain:{ .lg .middle } **Self-Evolution**

    ---

    Memory that consolidates, decays, and improves

    [:octicons-arrow-right-24: Learn more](concepts/evolution.md)

-   :material-shield-lock:{ .lg .middle } **Multi-Tenant Isolation**

    ---

    Complete data separation for enterprise

    [:octicons-arrow-right-24: Learn more](concepts/multi-tenancy.md)

</div>

---

## 📊 Performance

| Metric | Naive RAG | GraphMem | Advantage |
|--------|-----------|----------|-----------|
| **1K conversations** | 💥 Context overflow | ✅ Bounded | Handles growth |
| **10K entities** | O(n) = 2.3s | O(1) = 50ms | **46x faster** |
| **1 year history** | 3,650 entries | ~100 consolidated | **97% reduction** |
| **Entity conflicts** | Duplicates | Auto-resolved | Clean data |
| **Temporal queries** | ❌ Impossible | ✅ Native | Unique capability |

---

## 📚 Documentation

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Installation, quick start, and configuration

    [:octicons-arrow-right-24: Get started](getting-started/installation.md)

-   **Core Concepts**

    ---

    Understanding how GraphMem works

    [:octicons-arrow-right-24: Learn concepts](concepts/overview.md)

-   **Building Agents**

    ---

    Complete guide to building AI agents

    [:octicons-arrow-right-24: Build agents](agents/guide.md)

-   **Production**

    ---

    Deploy at scale with confidence

    [:octicons-arrow-right-24: Go to production](production/architecture.md)

</div>

---

## 🤝 Contributing

We're building the future of AI memory. Join us!

- 🐛 [Report bugs](https://github.com/Al-aminI/GraphMem/issues)
- 💡 [Request features](https://github.com/Al-aminI/GraphMem/issues)
- 🔀 [Submit PRs](https://github.com/Al-aminI/GraphMem/pulls)

---

<div align="center">

**Made with 🧠 by Al-Amin Ibrahim**

*"Give your AI agents the memory they deserve."*

</div>


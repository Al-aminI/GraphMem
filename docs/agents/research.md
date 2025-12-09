# Research Agents

Build agents that learn from multiple documents and answer complex questions.

## Pattern Overview

A research agent:
- Ingests multiple documents efficiently
- Synthesizes knowledge across sources
- Answers complex multi-hop questions
- Finds connections between entities

---

## Basic Implementation

```python
from graphmem import GraphMem, MemoryConfig

class ResearchAgent:
    """Agent that learns from multiple documents."""
    
    def __init__(self):
        # Option 1: OpenAI
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o",  # Use stronger model for research
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            
            # Use Neo4j for complex graph queries
            neo4j_uri="neo4j+s://...",
            neo4j_username="neo4j",
            neo4j_password="...",
            
            # Redis for caching
            redis_url="redis://...",
            
            # Aggressive evolution for research
            evolution_enabled=True,
            consolidation_threshold=0.75,
        )
        
        # Option 2: OpenRouter (with custom base URL)
        # self.config = MemoryConfig(
        #     llm_provider="openai_compatible",
        #     llm_api_key="sk-or-v1-...",
        #     llm_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        #     llm_model="google/gemini-2.0-flash-001",
        #     embedding_provider="openai_compatible",
        #     embedding_api_key="sk-or-v1-...",
        #     embedding_api_base="https://openrouter.ai/api/v1",  # Custom base URL
        #     embedding_model="openai/text-embedding-3-small",
        #     ...
        # )
        
        # Option 3: Azure OpenAI
        # self.config = MemoryConfig(
        #     llm_provider="azure_openai",
        #     llm_api_key="your-azure-key",
        #     llm_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
        #     azure_deployment="gpt-4",
        #     llm_model="gpt-4",
        #     azure_api_version="2024-02-15-preview",
        #     embedding_provider="azure_openai",
        #     embedding_api_key="your-azure-key",
        #     embedding_api_base="https://your-resource.openai.azure.com/",  # Azure endpoint
        #     azure_embedding_deployment="text-embedding-ada-002",
        #     embedding_model="text-embedding-ada-002",
        #     ...
        # )
        self.memory = GraphMem(self.config)
    
    def ingest_documents(self, documents: list[dict]):
        """Ingest multiple documents efficiently."""
        result = self.memory.ingest_batch(
            documents,
            max_workers=20,
            aggressive=True,
            show_progress=True,
        )
        
        print(f"Ingested {result['documents_processed']} docs")
        print(f"Extracted {result['total_entities']} entities")
        print(f"Found {result['total_relationships']} relationships")
        
        # Evolve after batch ingestion
        self.memory.evolve()
    
    def research(self, question: str) -> dict:
        """Answer complex research questions."""
        response = self.memory.query(question)
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [n.name for n in response.nodes[:5]],
            "related_entities": [n.name for n in response.nodes],
            "context_tokens": len(response.context.split()),
        }
    
    def find_connections(self, entity_a: str, entity_b: str) -> str:
        """Find how two entities are connected."""
        query = f"How are {entity_a} and {entity_b} related?"
        response = self.memory.query(query)
        return response.answer
    
    def summarize_topic(self, topic: str) -> str:
        """Summarize everything known about a topic."""
        query = f"Summarize everything known about {topic}"
        response = self.memory.query(query)
        return response.answer
```

---

## Usage Example

```python
agent = ResearchAgent()

# Ingest research papers
papers = [
    {"id": "paper1", "content": open("transformer_paper.txt").read()},
    {"id": "paper2", "content": open("attention_paper.txt").read()},
    {"id": "paper3", "content": open("gpt_paper.txt").read()},
    {"id": "paper4", "content": open("bert_paper.txt").read()},
]
agent.ingest_documents(papers)

# Ask complex questions
result = agent.research(
    "How did attention mechanisms evolve into modern LLMs?"
)
print(result["answer"])

# Find connections
connection = agent.find_connections("Transformers", "GPT-4")
print(connection)

# Get topic summaries
summary = agent.summarize_topic("self-attention")
print(summary)
```

---

## Multi-Source Research

### Different Document Types

```python
class MultiSourceResearchAgent(ResearchAgent):
    
    def ingest_paper(self, content: str, metadata: dict):
        """Ingest academic paper with structured metadata."""
        enriched = f"""
        ACADEMIC PAPER
        Title: {metadata.get('title', 'Unknown')}
        Authors: {', '.join(metadata.get('authors', []))}
        Year: {metadata.get('year', 'Unknown')}
        
        Abstract:
        {metadata.get('abstract', '')}
        
        Content:
        {content}
        """
        self.memory.ingest(enriched, importance=MemoryImportance.HIGH)
    
    def ingest_news(self, content: str, metadata: dict):
        """Ingest news article with temporal context."""
        enriched = f"""
        NEWS ARTICLE (Date: {metadata.get('date', 'Unknown')})
        Source: {metadata.get('source', 'Unknown')}
        
        {content}
        """
        self.memory.ingest(enriched, importance=MemoryImportance.MEDIUM)
    
    def ingest_internal_doc(self, content: str, metadata: dict):
        """Ingest internal document as critical."""
        self.memory.ingest(content, importance=MemoryImportance.CRITICAL)
```

---

## Citation Tracking

```python
class CitationAwareAgent(ResearchAgent):
    
    def research_with_citations(self, question: str) -> dict:
        """Answer with source citations."""
        response = self.memory.query(question)
        
        # Extract source documents
        sources = []
        for node in response.nodes:
            if "source_document" in node.properties:
                sources.append({
                    "document": node.properties["source_document"],
                    "excerpt": node.properties.get("source_chunk", "")[:200],
                })
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "citations": sources[:5],
        }
```

---

## Comparative Analysis

```python
class ComparativeAgent(ResearchAgent):
    
    def compare(self, entity_a: str, entity_b: str, aspect: str) -> str:
        """Compare two entities on a specific aspect."""
        query = f"""
        Compare {entity_a} and {entity_b} in terms of {aspect}.
        What are the key differences and similarities?
        """
        return self.memory.query(query).answer
    
    def timeline(self, entity: str) -> str:
        """Get chronological timeline for an entity."""
        query = f"""
        Provide a chronological timeline of key events for {entity}.
        Include dates when available.
        """
        return self.memory.query(query).answer
```

---

## Best Practices

1. **Use batch ingestion** - Much faster than sequential
2. **Enrich with metadata** - Add structure to documents
3. **Evolve after ingestion** - Consolidate knowledge
4. **Use stronger LLMs** - Research benefits from GPT-4/Claude
5. **Set high importance** - Research docs shouldn't decay
6. **Track sources** - Include document IDs in metadata


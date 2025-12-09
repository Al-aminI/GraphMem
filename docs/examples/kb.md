# Knowledge Base Example

Build an enterprise knowledge base with GraphMem.

## Complete Implementation

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance
from datetime import datetime
import os

class KnowledgeBase:
    """Enterprise knowledge base with policies, procedures, and employees."""
    
    def __init__(self, org_id: str):
        self.config = MemoryConfig(
            # Azure OpenAI for enterprise
            llm_provider="azure",
            llm_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_deployment="gpt-4",
            llm_model="gpt-4",
            
            embedding_provider="azure",
            embedding_api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_embedding_deployment="text-embedding-ada-002",
            embedding_model="text-embedding-ada-002",
            
            # Production storage
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            redis_url=os.getenv("REDIS_URL"),
            
            # Evolution
            evolution_enabled=True,
        )
        self.memory = GraphMem(self.config, memory_id=f"kb_{org_id}", user_id=org_id)
        self.org_id = org_id
    
    def add_policy(
        self,
        name: str,
        content: str,
        effective_date: str,
        department: str = "All",
    ):
        """Add or update a policy document."""
        enriched = f"""
        POLICY DOCUMENT
        ===============
        Name: {name}
        Department: {department}
        Effective Date: {effective_date}
        Last Updated: {datetime.now().isoformat()}
        
        POLICY CONTENT:
        {content}
        """
        
        self.memory.ingest(
            enriched,
            metadata={
                "type": "policy",
                "name": name,
                "department": department,
                "effective_date": effective_date,
            },
            importance=MemoryImportance.CRITICAL,
        )
        
        # Resolve conflicts with old versions
        self.memory.evolve()
    
    def add_procedure(self, name: str, steps: list[str], department: str):
        """Add a procedure with step-by-step instructions."""
        content = f"""
        PROCEDURE: {name}
        Department: {department}
        Last Updated: {datetime.now().isoformat()}
        
        STEPS:
        """
        for i, step in enumerate(steps, 1):
            content += f"\n{i}. {step}"
        
        self.memory.ingest(
            content,
            metadata={"type": "procedure", "name": name, "department": department},
            importance=MemoryImportance.HIGH,
        )
    
    def add_employee(self, data: dict):
        """Add employee information."""
        content = f"""
        EMPLOYEE PROFILE
        ================
        Name: {data['name']}
        Title: {data['title']}
        Department: {data['department']}
        Email: {data['email']}
        Phone: {data.get('phone', 'N/A')}
        Manager: {data.get('manager', 'N/A')}
        Start Date: {data.get('start_date', 'N/A')}
        Location: {data.get('location', 'N/A')}
        Skills: {', '.join(data.get('skills', []))}
        """
        
        self.memory.ingest(
            content,
            metadata={"type": "employee", **data},
            importance=MemoryImportance.MEDIUM,
        )
    
    def add_faq(self, question: str, answer: str, category: str):
        """Add a FAQ entry."""
        content = f"""
        FAQ: {question}
        Category: {category}
        
        Answer: {answer}
        """
        
        self.memory.ingest(
            content,
            metadata={"type": "faq", "category": category},
            importance=MemoryImportance.HIGH,
        )
    
    def search(self, query: str, department: str = None) -> dict:
        """Search the knowledge base."""
        if department:
            query = f"[Department: {department}] {query}"
        
        response = self.memory.query(query)
        
        return {
            "answer": response.answer,
            "confidence": response.confidence,
            "sources": [
                {"name": n.name, "type": n.entity_type}
                for n in response.nodes[:5]
            ],
            "needs_escalation": response.confidence < 0.5,
        }
    
    def bulk_import(self, documents: list[dict]):
        """Bulk import documents from various sources."""
        self.memory.ingest_batch(
            documents,
            max_workers=20,
            aggressive=True,
            show_progress=True,
        )
        self.memory.evolve()
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return self.memory.get_stats()

# Usage Example
if __name__ == "__main__":
    kb = KnowledgeBase("acme_corp")
    
    # Add policies
    kb.add_policy(
        name="Remote Work Policy",
        content="""
        Employees may work remotely up to 3 days per week.
        Remote work requires manager approval.
        Employees must be available during core hours (10am-4pm).
        """,
        effective_date="2024-01-01",
        department="All",
    )
    
    # Add procedures
    kb.add_procedure(
        name="Expense Reimbursement",
        steps=[
            "Submit expense report within 30 days of expense",
            "Include receipts for all items over $25",
            "Get manager approval for items over $500",
            "Finance will process within 5 business days",
        ],
        department="Finance",
    )
    
    # Add employees
    kb.add_employee({
        "name": "Jane Smith",
        "title": "CEO",
        "department": "Executive",
        "email": "jane@acme.com",
        "skills": ["Leadership", "Strategy", "Finance"],
    })
    
    # Add FAQ
    kb.add_faq(
        question="What are the office hours?",
        answer="Office hours are 9am to 6pm, Monday through Friday.",
        category="General",
    )
    
    # Search
    result = kb.search("Can I work from home?")
    print(result["answer"])
    # → "Yes, employees may work remotely up to 3 days per week 
    #    with manager approval..."
    
    result = kb.search("How do I submit expenses?")
    print(result["answer"])
    # → "Submit expense report within 30 days, include receipts 
    #    for items over $25..."
```

## FastAPI Integration

```python
from fastapi import FastAPI, Depends, Header

app = FastAPI()

# Singleton KB per org
kbs = {}

def get_kb(org_id: str = Header(..., alias="X-Org-ID")) -> KnowledgeBase:
    if org_id not in kbs:
        kbs[org_id] = KnowledgeBase(org_id)
    return kbs[org_id]

@app.post("/policy")
async def add_policy(
    name: str,
    content: str,
    effective_date: str,
    kb: KnowledgeBase = Depends(get_kb),
):
    kb.add_policy(name, content, effective_date)
    return {"status": "added"}

@app.post("/search")
async def search(
    query: str,
    department: str = None,
    kb: KnowledgeBase = Depends(get_kb),
):
    return kb.search(query, department)

@app.get("/stats")
async def stats(kb: KnowledgeBase = Depends(get_kb)):
    return kb.get_stats()
```


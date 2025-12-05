#!/usr/bin/env python3
"""
Synthetic Benchmark Generator for GraphMem

Generates challenging datasets that specifically test GraphMem's unique capabilities
where Naive RAG CANNOT succeed:

1. ENTITY RESOLUTION CHALLENGE
   - Same entity with different names/aliases across documents
   - Naive RAG retrieves wrong chunks due to name mismatch
   - GraphMem consolidates entities → correct answer

2. TEMPORAL CONFLICT CHALLENGE  
   - Facts that CHANGE over time (CEO changes, acquisitions)
   - Old and new facts in different documents
   - Query: "Who is the CURRENT CEO?"
   - Naive RAG retrieves old doc; GraphMem uses temporal validity

3. RELATIONSHIP CHAIN CHALLENGE
   - Multi-hop: A→B→C→D→E (5+ hops)
   - Information spread across 20+ documents
   - No single doc has full answer
   - GraphMem's knowledge graph traverses the path

4. SCALE + NOISE CHALLENGE
   - 500+ documents, only 5 are relevant
   - 95% noise documents on similar topics
   - Naive RAG's top-k retrieves noise
   - GraphMem's entity graph finds the signal

5. CONTRADICTION RESOLUTION CHALLENGE
   - Multiple sources with conflicting info
   - Need recency + source credibility
   - GraphMem's decay + importance scoring resolves

Usage:
    python synthetic_benchmark.py \
        --challenge entity_resolution \
        --num-documents 100 \
        --output synthetic_data/entity_resolution.json
"""

import argparse
import json
import logging
import os
import random
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class Document:
    """A synthetic document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""  # ISO format
    is_noise: bool = False


@dataclass
class Query:
    """A query with ground truth."""
    id: str
    question: str
    answer: str
    reasoning: str  # Why naive RAG fails
    required_hops: int = 1
    required_docs: List[str] = field(default_factory=list)
    challenge_type: str = ""


@dataclass
class Challenge:
    """A complete challenge dataset."""
    name: str
    description: str
    documents: List[Document]
    queries: List[Query]
    why_naive_rag_fails: str
    why_graphmem_succeeds: str


# ============================================================================
# GENERATORS
# ============================================================================
class SyntheticGenerator:
    """Base generator with LLM support."""
    
    def __init__(self, llm_provider=None):
        self.llm = llm_provider
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate content using LLM."""
        if self.llm:
            try:
                return self.llm.chat([{"role": "user", "content": prompt}])
            except:
                pass
        return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when no LLM available."""
        return ""
    
    def _random_name(self) -> str:
        """Generate a random person name."""
        first = random.choice(["James", "Sarah", "Michael", "Emily", "David", "Lisa", "Robert", "Jennifer", 
                               "William", "Amanda", "Thomas", "Jessica", "Daniel", "Ashley", "Christopher",
                               "Sophia", "Alexander", "Olivia", "Benjamin", "Emma"])
        last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                              "Thomas", "Taylor", "Moore", "Jackson", "Martin"])
        return f"{first} {last}"
    
    def _random_company(self) -> str:
        """Generate a random company name."""
        prefix = random.choice(["Quantum", "Apex", "Nova", "Vertex", "Nexus", "Zenith", "Prism", "Fusion",
                                "Stellar", "Vortex", "Helix", "Cipher", "Pulse", "Axis", "Orbit"])
        suffix = random.choice(["Technologies", "Systems", "Dynamics", "Industries", "Solutions", 
                                "Innovations", "Labs", "Corp", "Inc", "Global", "AI", "Tech"])
        return f"{prefix} {suffix}"


class EntityResolutionGenerator(SyntheticGenerator):
    """
    CHALLENGE 1: Entity Resolution
    
    Same entity appears with different names/aliases across documents.
    Naive RAG fails because query uses one name, docs use different names.
    GraphMem consolidates entities and finds the connection.
    """
    
    def generate(self, num_docs: int = 50) -> Challenge:
        logger.info("Generating Entity Resolution Challenge...")
        
        # Create a central entity with many aliases
        real_name = "Alexander Chen"
        aliases = [
            "Alex Chen",
            "A. Chen", 
            "Dr. Chen",
            "Professor Chen",
            "Alexander C.",
            "Chen, Alexander",
            "AC",  # Initials
            "The Quantum Pioneer",  # Nickname
        ]
        
        # Facts about this person (spread across docs with different aliases)
        facts = [
            ("founded Quantum AI Labs in 2015", "2015-03-15"),
            ("invented the neural bridge architecture", "2018-07-22"),
            ("won the Turing Award in 2022", "2022-12-10"),
            ("currently serves as Chief Scientist at Google DeepMind", "2023-06-01"),
            ("published 127 papers on machine learning", "2023-09-15"),
            ("holds 45 patents in quantum computing", "2023-01-20"),
            ("graduated from MIT with PhD in Computer Science", "2008-05-20"),
            ("previously worked at IBM Research for 7 years", "2008-06-01"),
            ("married to Dr. Sarah Chen, a biologist", "2010-08-14"),
            ("has two children and lives in Palo Alto", "2023-01-01"),
        ]
        
        documents = []
        
        # Create documents, each using a DIFFERENT alias
        for i, (fact, date) in enumerate(facts):
            alias = random.choice(aliases)
            
            # Create a document that uses THIS alias
            doc_content = self._create_entity_doc(alias, fact, i)
            
            documents.append(Document(
                id=f"entity_doc_{i}",
                content=doc_content,
                metadata={"alias_used": alias, "fact": fact},
                timestamp=date,
            ))
        
        # Add noise documents about OTHER people named Chen
        noise_chens = ["Michael Chen", "Linda Chen", "David Chen", "Jennifer Chen"]
        for i, noise_name in enumerate(noise_chens):
            noise_facts = [
                f"{noise_name} is a software engineer at Microsoft.",
                f"{noise_name} works in the Seattle office.",
                f"{noise_name} has been with the company for 5 years.",
            ]
            documents.append(Document(
                id=f"noise_chen_{i}",
                content=" ".join(noise_facts),
                metadata={"is_noise": True},
                is_noise=True,
            ))
        
        # Add more general noise
        for i in range(num_docs - len(documents)):
            documents.append(self._create_noise_doc(i))
        
        # Create queries that use the REAL name (not aliases)
        queries = [
            Query(
                id="er_q1",
                question="What company did Alexander Chen found?",
                answer="Quantum AI Labs",
                reasoning="Docs use aliases like 'Dr. Chen', 'The Quantum Pioneer'. Naive RAG searches for 'Alexander Chen' and finds nothing. GraphMem resolves aliases to same entity.",
                required_hops=1,
                required_docs=["entity_doc_0"],
                challenge_type="entity_resolution",
            ),
            Query(
                id="er_q2", 
                question="What awards has Alexander Chen won?",
                answer="Turing Award in 2022",
                reasoning="Award doc uses 'Professor Chen'. Query uses 'Alexander Chen'. Naive RAG misses it.",
                required_hops=1,
                required_docs=["entity_doc_2"],
                challenge_type="entity_resolution",
            ),
            Query(
                id="er_q3",
                question="Where does Alexander Chen currently work and what is his role?",
                answer="Chief Scientist at Google DeepMind",
                reasoning="Multiple docs needed. Each uses different alias. GraphMem consolidates.",
                required_hops=2,
                required_docs=["entity_doc_3"],
                challenge_type="entity_resolution",
            ),
            Query(
                id="er_q4",
                question="What is Alexander Chen's educational background and early career?",
                answer="PhD from MIT in Computer Science, then worked at IBM Research for 7 years",
                reasoning="Info spread across 2 docs with different aliases. Naive RAG can't connect them.",
                required_hops=2,
                required_docs=["entity_doc_6", "entity_doc_7"],
                challenge_type="entity_resolution",
            ),
        ]
        
        return Challenge(
            name="Entity Resolution Challenge",
            description="Same person appears with 8 different aliases across documents",
            documents=documents,
            queries=queries,
            why_naive_rag_fails="Query uses 'Alexander Chen' but documents use aliases like 'Dr. Chen', 'The Quantum Pioneer', 'A. Chen'. Embedding similarity fails because names don't match.",
            why_graphmem_succeeds="Entity resolution consolidates all aliases into a single entity node. Query on any name variant finds the consolidated entity with all facts.",
        )
    
    def _create_entity_doc(self, alias: str, fact: str, idx: int) -> str:
        """Create a document about the entity using specific alias."""
        templates = [
            f"In a recent interview, {alias} discussed how they {fact}. The accomplishment was celebrated worldwide.",
            f"According to sources, {alias} {fact}. This marks a significant milestone in their career.",
            f"{alias} has {fact}, according to official records. The announcement was made at a conference.",
            f"Industry insiders confirm that {alias} {fact}. The development has implications for the field.",
            f"Reports indicate {alias} {fact}. Colleagues praised the achievement.",
        ]
        return random.choice(templates)
    
    def _create_noise_doc(self, idx: int) -> Document:
        """Create a noise document about unrelated topics."""
        topics = [
            "The latest advances in renewable energy show promising results for solar panel efficiency.",
            "Stock markets fluctuated today as investors reacted to economic indicators.",
            "New research suggests that Mediterranean diet may improve cognitive function.",
            "Climate scientists report accelerating ice melt in polar regions.",
            "Tech companies are investing heavily in quantum computing research.",
        ]
        return Document(
            id=f"noise_{idx}",
            content=random.choice(topics) + f" Document reference: {idx}.",
            is_noise=True,
        )


class TemporalConflictGenerator(SyntheticGenerator):
    """
    CHALLENGE 2: Temporal Conflicts
    
    Facts change over time. Multiple documents have different (outdated) info.
    Query asks for CURRENT state. Naive RAG may retrieve old docs.
    GraphMem's temporal validity (valid_from/valid_until) resolves.
    """
    
    def generate(self, num_docs: int = 50) -> Challenge:
        logger.info("Generating Temporal Conflict Challenge...")
        
        company = "NovaTech Industries"
        
        # CEO history (most recent is current)
        ceo_history = [
            ("John Mitchell", "2010-01-15", "2015-06-30", "stepped down for health reasons"),
            ("Sarah Williams", "2015-07-01", "2018-12-31", "left to start own company"),
            ("David Park", "2019-01-01", "2021-08-15", "retired after successful tenure"),
            ("Emily Zhang", "2021-08-16", "2023-03-01", "moved to board position"),
            ("Marcus Johnson", "2023-03-02", None, "current CEO"),  # CURRENT
        ]
        
        documents = []
        
        # Create documents for EACH CEO tenure (all look authoritative)
        for i, (ceo, start, end, reason) in enumerate(ceo_history):
            # Create multiple docs per CEO to make it harder
            for j in range(3):
                if end:
                    doc_content = f"""
                    {company} Leadership Update ({start})
                    
                    {ceo} has been appointed as the new CEO of {company}. 
                    The board unanimously approved the appointment.
                    {ceo} brings extensive experience to the role and will lead
                    the company's strategic initiatives. As CEO, {ceo} will
                    oversee all operations and report directly to the board.
                    """
                else:
                    doc_content = f"""
                    {company} Executive Announcement ({start})
                    
                    {company} is pleased to announce that {ceo} has assumed
                    the role of Chief Executive Officer. {ceo} previously served
                    as COO and brings deep knowledge of the company. As the
                    current CEO, {ceo} will drive the company's vision forward.
                    """
                
                documents.append(Document(
                    id=f"ceo_doc_{i}_{j}",
                    content=doc_content.strip(),
                    metadata={"ceo": ceo, "start": start, "end": end},
                    timestamp=start,
                ))
        
        # Add more historical docs that mention OLD CEOs as "the CEO"
        for i, (ceo, start, end, reason) in enumerate(ceo_history[:-1]):  # Exclude current
            docs_content = [
                f"In {start[:4]}, {ceo}, CEO of {company}, announced record profits.",
                f"CEO {ceo} spoke at the industry conference about {company}'s strategy.",
                f"Under {ceo}'s leadership as CEO, {company} expanded to 15 countries.",
            ]
            for j, content in enumerate(docs_content):
                documents.append(Document(
                    id=f"historical_{i}_{j}",
                    content=content,
                    metadata={"ceo": ceo, "historical": True},
                    timestamp=start,
                ))
        
        # Add noise
        for i in range(num_docs - len(documents)):
            documents.append(Document(
                id=f"noise_{i}",
                content=f"Industry news: Technology sector shows growth. Reference {i}.",
                is_noise=True,
            ))
        
        queries = [
            Query(
                id="tc_q1",
                question=f"Who is the CURRENT CEO of {company}?",
                answer="Marcus Johnson",
                reasoning="There are 15+ docs mentioning different CEOs. Naive RAG may retrieve older docs about Sarah Williams or David Park. GraphMem's temporal validity knows only Marcus Johnson is CURRENT.",
                required_hops=1,
                challenge_type="temporal_conflict",
            ),
            Query(
                id="tc_q2",
                question=f"Who was the CEO of {company} before the current one?",
                answer="Emily Zhang",
                reasoning="Requires understanding temporal sequence. Naive RAG can't order by time.",
                required_hops=2,
                challenge_type="temporal_conflict",
            ),
            Query(
                id="tc_q3",
                question=f"How many CEOs has {company} had and who are they in order?",
                answer="5 CEOs: John Mitchell, Sarah Williams, David Park, Emily Zhang, Marcus Johnson",
                reasoning="Requires assembling temporal chain from scattered docs. Naive RAG retrieves random subset.",
                required_hops=5,
                challenge_type="temporal_conflict",
            ),
        ]
        
        return Challenge(
            name="Temporal Conflict Challenge",
            description="Company has had 5 CEOs. Query asks for CURRENT one.",
            documents=documents,
            queries=queries,
            why_naive_rag_fails="15+ documents mention different people as 'CEO'. Naive RAG retrieves by text similarity, often returning old announcements. No way to know which is current.",
            why_graphmem_succeeds="Temporal validity (valid_from/valid_until) on relationships. When evolve() runs, old CEO relationships get valid_until set. Query for 'current' only returns active relationships.",
        )


class RelationshipChainGenerator(SyntheticGenerator):
    """
    CHALLENGE 3: Multi-Hop Relationship Chains
    
    Answer requires traversing: A → B → C → D → E (5+ hops)
    Each hop is in a DIFFERENT document.
    No single document has enough info.
    Naive RAG's top-k can't get all needed docs.
    GraphMem traverses the knowledge graph.
    """
    
    def generate(self, num_docs: int = 100, chain_length: int = 6) -> Challenge:
        logger.info(f"Generating {chain_length}-Hop Relationship Chain Challenge...")
        
        # Create a chain of people with relationships
        chain = []
        for i in range(chain_length):
            chain.append({
                "name": self._random_name(),
                "company": self._random_company(),
                "role": random.choice(["CEO", "CTO", "CFO", "VP Engineering", "Director"]),
            })
        
        # The secret: Last person in chain has crucial info
        secret_info = {
            "project": "Project Prometheus",
            "budget": "$50 million",
            "location": "Geneva, Switzerland",
            "deadline": "December 2025",
        }
        
        documents = []
        
        # Create ONE document per relationship hop
        # Each doc ONLY connects adjacent people
        for i in range(len(chain) - 1):
            person_a = chain[i]
            person_b = chain[i + 1]
            
            relationship = random.choice([
                f"reports directly to",
                f"works closely with",
                f"was mentored by",
                f"frequently collaborates with",
            ])
            
            doc_content = f"""
            Internal Memo - Confidential
            
            {person_a['name']} ({person_a['role']} at {person_a['company']})
            {relationship} {person_b['name']}.
            
            Their collaboration has been productive.
            For questions about ongoing projects, contact {person_b['name']}.
            """
            
            documents.append(Document(
                id=f"chain_doc_{i}",
                content=doc_content.strip(),
                metadata={"hop": i, "connects": [person_a['name'], person_b['name']]},
            ))
        
        # Final document: Last person has the secret info
        final_person = chain[-1]
        final_doc = f"""
        Project Update - {final_person['name']}
        
        {final_person['name']} ({final_person['role']} at {final_person['company']})
        is leading {secret_info['project']}.
        
        Budget: {secret_info['budget']}
        Location: {secret_info['location']}
        Target Deadline: {secret_info['deadline']}
        
        This is classified information.
        """
        
        documents.append(Document(
            id="chain_final",
            content=final_doc.strip(),
            metadata={"has_secret": True},
        ))
        
        # Add LOTS of noise (similar corporate docs)
        for i in range(num_docs - len(documents)):
            noise_person = self._random_name()
            noise_company = self._random_company()
            noise_content = f"""
            {noise_person} works at {noise_company} as a {random.choice(['Manager', 'Director', 'VP'])}.
            They are involved in various projects and initiatives.
            Contact them for more information about {noise_company}'s activities.
            """
            documents.append(Document(
                id=f"noise_{i}",
                content=noise_content.strip(),
                is_noise=True,
            ))
        
        # Shuffle to hide the chain
        random.shuffle(documents)
        
        first_person = chain[0]
        queries = [
            Query(
                id="rc_q1",
                question=f"What project is connected to {first_person['name']}?",
                answer=f"{secret_info['project']}",
                reasoning=f"Requires {chain_length} hops: {' → '.join([p['name'] for p in chain])}. Each hop is in a different doc. Naive RAG top-5 retrieves docs about {first_person['name']} but not the chain.",
                required_hops=chain_length,
                required_docs=[f"chain_doc_{i}" for i in range(chain_length-1)] + ["chain_final"],
                challenge_type="relationship_chain",
            ),
            Query(
                id="rc_q2",
                question=f"What is the budget of the project that {first_person['name']} is indirectly connected to?",
                answer=secret_info['budget'],
                reasoning=f"Must traverse full chain to find project, then get budget. {chain_length} hops minimum.",
                required_hops=chain_length,
                challenge_type="relationship_chain",
            ),
            Query(
                id="rc_q3",
                question=f"Who is leading the project and where is it located?",
                answer=f"{final_person['name']} in {secret_info['location']}",
                reasoning="Must find the project first (via chain), then get leader and location.",
                required_hops=chain_length + 1,
                challenge_type="relationship_chain",
            ),
        ]
        
        return Challenge(
            name=f"{chain_length}-Hop Relationship Chain Challenge",
            description=f"Answer requires traversing {chain_length} relationship hops across {chain_length+1} documents",
            documents=documents,
            queries=queries,
            why_naive_rag_fails=f"Query mentions first person. Top-k retrieval gets docs about them, but answer is {chain_length} hops away. No single doc has the answer. Top-5 or top-10 can't span the full chain.",
            why_graphmem_succeeds=f"Knowledge graph has edges: A→B→C→D→E→F. Query starts at A, graph traversal reaches F with the answer. Entity relationships enable multi-hop reasoning.",
        )


class ScaleNoiseGenerator(SyntheticGenerator):
    """
    CHALLENGE 4: Scale + Noise
    
    500+ documents, only 5 are relevant.
    95% are noise on SIMILAR topics (to fool embeddings).
    Naive RAG's top-k retrieves mostly noise.
    GraphMem's entity graph finds signal.
    """
    
    def generate(self, num_docs: int = 500, num_relevant: int = 5) -> Challenge:
        logger.info(f"Generating Scale Challenge: {num_docs} docs, only {num_relevant} relevant...")
        
        # The target entity
        target_company = "Helix Quantum Computing"
        target_ceo = "Dr. Amara Okonkwo"
        
        # Relevant facts (spread across num_relevant docs)
        relevant_facts = [
            f"{target_company} was founded in 2019 in Boston.",
            f"{target_ceo} serves as CEO of {target_company}.",
            f"{target_company} raised $200 million in Series C funding.",
            f"{target_company}'s quantum processor achieved 1000 qubit milestone.",
            f"{target_ceo} has a PhD from Stanford in Quantum Physics.",
        ]
        
        documents = []
        
        # Create relevant documents
        for i, fact in enumerate(relevant_facts):
            documents.append(Document(
                id=f"relevant_{i}",
                content=f"{fact} This achievement was recognized by industry experts.",
                metadata={"is_relevant": True},
            ))
        
        # Create NOISE documents about OTHER quantum computing companies
        noise_companies = [
            "Quantum Dynamics Corp", "QBit Systems", "Entangle Tech",
            "Superposition Labs", "Wave Function Inc", "Qubit Industries",
            "Coherence Computing", "Quantum Logic Systems", "Phase Space Tech",
            "Probability Wave Co", "Quantum Tunneling Ltd", "Spin State Computing",
        ]
        
        noise_ceos = [
            "Dr. James Smith", "Dr. Maria Garcia", "Dr. Wei Chen",
            "Dr. Robert Johnson", "Dr. Emma Wilson", "Dr. Ahmed Hassan",
        ]
        
        # Generate noise that looks SIMILAR to relevant docs
        for i in range(num_docs - num_relevant):
            noise_company = random.choice(noise_companies)
            noise_ceo = random.choice(noise_ceos)
            
            noise_templates = [
                f"{noise_company} announced new quantum computing breakthrough today.",
                f"CEO {noise_ceo} of {noise_company} spoke at the quantum computing summit.",
                f"{noise_company} raised funding for quantum processor development.",
                f"Industry leader {noise_company} expands quantum research facility.",
                f"{noise_ceo} leads {noise_company}'s push into quantum supremacy.",
                f"Quantum computing startup {noise_company} hires 50 new researchers.",
                f"{noise_company}'s quantum chip shows promising results in tests.",
                f"Boston-based quantum firm announces partnership with universities.",  # Same city!
                f"New quantum computing company secures Series C funding.",  # Same funding round!
            ]
            
            documents.append(Document(
                id=f"noise_{i}",
                content=random.choice(noise_templates),
                is_noise=True,
            ))
        
        # Shuffle
        random.shuffle(documents)
        
        queries = [
            Query(
                id="sn_q1",
                question=f"Who is the CEO of {target_company}?",
                answer=target_ceo,
                reasoning=f"Only 1 of {num_docs} docs mentions {target_company}. Other docs mention similar quantum companies. Embedding similarity retrieves noise about 'quantum computing' and 'CEO'.",
                required_hops=1,
                required_docs=["relevant_1"],
                challenge_type="scale_noise",
            ),
            Query(
                id="sn_q2",
                question=f"How much funding did {target_company} raise?",
                answer="$200 million in Series C",
                reasoning="Many noise docs mention 'funding' and 'Series C'. Embeddings can't distinguish target company.",
                required_hops=1,
                required_docs=["relevant_2"],
                challenge_type="scale_noise",
            ),
            Query(
                id="sn_q3",
                question=f"What technical milestone did {target_company} achieve?",
                answer="1000 qubit quantum processor",
                reasoning="Many noise docs mention 'quantum processor' and 'milestone'. Only 1 doc is about target.",
                required_hops=1,
                required_docs=["relevant_3"],
                challenge_type="scale_noise",
            ),
        ]
        
        return Challenge(
            name=f"Scale + Noise Challenge ({num_docs} docs)",
            description=f"{num_docs} documents about quantum computing, only {num_relevant} about target company",
            documents=documents,
            queries=queries,
            why_naive_rag_fails=f"Query about '{target_company}' retrieves docs with similar embeddings (quantum computing, CEO, funding). 95%+ of top-k results are noise about other companies.",
            why_graphmem_succeeds=f"Entity graph has node for '{target_company}' with edges to its specific facts. Query retrieves the entity and its relationships, not random similar text.",
        )


# ============================================================================
# EVALUATOR
# ============================================================================
class SyntheticBenchmarkEvaluator:
    """Evaluate GraphMem vs Naive RAG on synthetic challenges."""
    
    def __init__(
        self,
        provider: str = "azure",
        api_key: str = None,
        api_base: str = None,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        azure_deployment: str = None,
        azure_embedding_deployment: str = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.azure_deployment = azure_deployment or llm_model
        self.azure_embedding_deployment = azure_embedding_deployment or embedding_model
    
    def _init_graphmem(self, db_path: str = "synthetic_bench.db"):
        """Initialize GraphMem."""
        from graphmem import GraphMem, MemoryConfig
        
        if os.path.exists(db_path):
            os.remove(db_path)
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        config = MemoryConfig(
            llm_provider=provider_name,
            llm_api_key=self.api_key,
            llm_api_base=self.api_base,
            llm_model=self.llm_model,
            embedding_provider=provider_name,
            embedding_api_key=self.api_key,
            embedding_api_base=self.api_base,
            embedding_model=self.embedding_model,
            azure_api_version="2024-08-01-preview",
            azure_deployment=self.azure_deployment,
            azure_embedding_deployment=self.azure_embedding_deployment,
            turso_db_path=db_path,
        )
        
        return GraphMem(config)
    
    def _init_naive_rag(self):
        """Initialize simple RAG baseline."""
        from graphmem.llm.providers import get_llm_provider
        from graphmem.llm.embeddings import get_embedding_provider
        import numpy as np
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        
        llm = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,
        )
        
        embedder = get_embedding_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.embedding_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_embedding_deployment,
        )
        
        class SimpleRAG:
            def __init__(self, llm, embedder):
                self.llm = llm
                self.embedder = embedder
                self.chunks = []
                self.embeddings = []
            
            def ingest(self, docs):
                for doc in docs:
                    content = doc.content
                    try:
                        emb = self.embedder.embed_text(content)
                        self.chunks.append(content)
                        self.embeddings.append(emb)
                    except:
                        pass
            
            def query(self, question, top_k=5):
                if not self.embeddings:
                    return "No documents."
                
                q_emb = self.embedder.embed_text(question)
                
                # Cosine similarity
                sims = []
                for i, emb in enumerate(self.embeddings):
                    a, b = np.array(q_emb), np.array(emb)
                    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    sims.append((i, sim))
                
                sims.sort(key=lambda x: x[1], reverse=True)
                
                context = "\n\n".join([self.chunks[i] for i, _ in sims[:top_k]])
                
                prompt = f"Answer based on context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
                return self.llm.chat([{"role": "user", "content": prompt}])
        
        return SimpleRAG(llm, embedder)
    
    def evaluate(self, challenge: Challenge) -> Dict:
        """Evaluate both systems on a challenge."""
        print(f"\n{'='*80}")
        print(f"🧪 EVALUATING: {challenge.name}")
        print(f"   Documents: {len(challenge.documents)}")
        print(f"   Queries: {len(challenge.queries)}")
        print(f"{'='*80}")
        
        print(f"\n❌ Why Naive RAG Fails:")
        print(f"   {challenge.why_naive_rag_fails}")
        print(f"\n✅ Why GraphMem Succeeds:")
        print(f"   {challenge.why_graphmem_succeeds}")
        
        # Initialize systems
        gm = self._init_graphmem()
        rag = self._init_naive_rag()
        
        # Ingest documents
        print(f"\n📥 Ingesting {len(challenge.documents)} documents...")
        
        docs_for_batch = [{"id": d.id, "content": d.content} for d in challenge.documents]
        
        print("   GraphMem (with entity resolution + temporal validity)...")
        gm.ingest_batch(documents=docs_for_batch, max_workers=20, aggressive=True)
        gm.evolve()  # Consolidate entities, build relationships
        
        print("   Naive RAG (chunk + embed)...")
        rag.ingest(challenge.documents)
        
        # Query and compare
        print(f"\n🔍 Running {len(challenge.queries)} queries...")
        
        gm_correct = 0
        rag_correct = 0
        
        for q in challenge.queries:
            print(f"\n   Q: {q.question[:60]}...")
            print(f"   Expected: {q.answer}")
            print(f"   Required hops: {q.required_hops}")
            
            # GraphMem
            try:
                gm_response = gm.query(q.question)
                gm_answer = gm_response.answer
            except Exception as e:
                gm_answer = f"Error: {e}"
            
            # Naive RAG
            try:
                rag_answer = rag.query(q.question)
            except Exception as e:
                rag_answer = f"Error: {e}"
            
            # Check correctness (simple contains check)
            gm_is_correct = q.answer.lower() in gm_answer.lower()
            rag_is_correct = q.answer.lower() in rag_answer.lower()
            
            if gm_is_correct:
                gm_correct += 1
            if rag_is_correct:
                rag_correct += 1
            
            print(f"   GraphMem: {'✅' if gm_is_correct else '❌'} {gm_answer[:60]}...")
            print(f"   NaiveRAG: {'✅' if rag_is_correct else '❌'} {rag_answer[:60]}...")
        
        # Results
        print(f"\n{'='*80}")
        print("📊 RESULTS")
        print(f"{'='*80}")
        print(f"   GraphMem: {gm_correct}/{len(challenge.queries)} ({100*gm_correct/len(challenge.queries):.0f}%)")
        print(f"   NaiveRAG: {rag_correct}/{len(challenge.queries)} ({100*rag_correct/len(challenge.queries):.0f}%)")
        
        if gm_correct > rag_correct:
            print(f"\n   🏆 GraphMem WINS by {gm_correct - rag_correct} questions!")
        elif rag_correct > gm_correct:
            print(f"\n   📚 NaiveRAG wins by {rag_correct - gm_correct} questions")
        else:
            print(f"\n   🤝 Tie")
        
        return {
            "challenge": challenge.name,
            "graphmem_correct": gm_correct,
            "naive_rag_correct": rag_correct,
            "total_queries": len(challenge.queries),
        }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Synthetic Benchmark for GraphMem")
    
    parser.add_argument("--challenge", default="all", 
                        choices=["entity_resolution", "temporal_conflict", "relationship_chain", "scale_noise", "all"])
    parser.add_argument("--num-documents", type=int, default=50)
    parser.add_argument("--chain-length", type=int, default=6)
    
    parser.add_argument("--provider", default="azure")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--azure-endpoint", help="Azure endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4.1-mini")
    parser.add_argument("--azure-embedding-deployment", default="text-embedding-3-small")
    
    parser.add_argument("--output", default="synthetic_results.json")
    parser.add_argument("--generate-only", action="store_true", help="Only generate data, don't evaluate")
    
    args = parser.parse_args()
    
    # Generate challenges
    challenges = []
    
    if args.challenge in ["entity_resolution", "all"]:
        challenges.append(EntityResolutionGenerator().generate(args.num_documents))
    
    if args.challenge in ["temporal_conflict", "all"]:
        challenges.append(TemporalConflictGenerator().generate(args.num_documents))
    
    if args.challenge in ["relationship_chain", "all"]:
        challenges.append(RelationshipChainGenerator().generate(args.num_documents, args.chain_length))
    
    if args.challenge in ["scale_noise", "all"]:
        challenges.append(ScaleNoiseGenerator().generate(max(args.num_documents, 200)))
    
    if args.generate_only:
        # Just save the generated data
        output = {
            "generated_at": datetime.now().isoformat(),
            "challenges": [asdict(c) for c in challenges],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Saved {len(challenges)} challenges to {args.output}")
        return
    
    # Evaluate
    evaluator = SyntheticBenchmarkEvaluator(
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.azure_endpoint,
        llm_model=args.azure_deployment,
        embedding_model=args.azure_embedding_deployment,
        azure_deployment=args.azure_deployment,
        azure_embedding_deployment=args.azure_embedding_deployment,
    )
    
    all_results = []
    for challenge in challenges:
        result = evaluator.evaluate(challenge)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_gm = sum(r["graphmem_correct"] for r in all_results)
    total_rag = sum(r["naive_rag_correct"] for r in all_results)
    total_q = sum(r["total_queries"] for r in all_results)
    
    print(f"\n{'Challenge':<40} {'GraphMem':>15} {'NaiveRAG':>15}")
    print("-" * 70)
    for r in all_results:
        gm = f"{r['graphmem_correct']}/{r['total_queries']}"
        rag = f"{r['naive_rag_correct']}/{r['total_queries']}"
        print(f"{r['challenge']:<40} {gm:>15} {rag:>15}")
    print("-" * 70)
    print(f"{'TOTAL':<40} {f'{total_gm}/{total_q}':>15} {f'{total_rag}/{total_q}':>15}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "results": all_results,
            "total_graphmem": total_gm,
            "total_naive_rag": total_rag,
            "total_queries": total_q,
        }, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()


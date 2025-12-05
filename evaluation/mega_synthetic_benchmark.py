#!/usr/bin/env python3
"""
MEGA Synthetic Benchmark Generator for GraphMem

Uses LLM to generate MASSIVE, realistic conversation-style data that:
1. BREAKS naive RAG context window (too much data)
2. Simulates real conversations with evolving facts
3. Has entity name variations throughout
4. Includes massive noise that fools embeddings
5. Requires multi-hop reasoning across 1000+ messages

The goal: Create a benchmark that is IMPOSSIBLE for naive RAG to solve
because the answer requires reasoning over data that exceeds context limits.

Challenges:
1. CONVERSATION EVOLUTION (1000+ turns)
   - Long conversation with facts changing over time
   - Person changes jobs 5 times, relationships evolve
   - Query: "What is John's current job?" requires finding the LATEST
   - Naive RAG context: 8K-128K tokens, conversation: 500K+ tokens

2. ENTERPRISE KNOWLEDGE BASE (5000+ documents)
   - Company wiki with interconnected facts
   - Same entity mentioned in 100+ documents with variations
   - Naive RAG top-10 retrieves 10 of 5000 docs (0.2%)

3. TEMPORAL CHRONICLE (Years of data)
   - News articles spanning 5 years
   - Facts change: CEO, stock price, acquisitions
   - Query about "current state" requires temporal reasoning

Usage:
    python mega_synthetic_benchmark.py \
        --provider azure \
        --api-key "..." \
        --scenario conversation_evolution \
        --num-turns 1000
"""

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# LLM-POWERED GENERATION
# ============================================================================
class LLMGenerator:
    """LLM-powered content generator."""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.cache = {}
    
    def generate(self, prompt: str, temperature: float = 0.8) -> str:
        """Generate content using LLM with caching."""
        cache_key = hash(prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature
            )
            self.cache[cache_key] = response
            return response
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""
    
    def generate_person_profile(self) -> Dict:
        """Generate a detailed person profile."""
        prompt = """Generate a fictional person profile as JSON:
{
    "name": "Full Name",
    "nicknames": ["Nick1", "Nick2", "Professional Title"],
    "current_job": {"title": "...", "company": "...", "since": "2023-01"},
    "job_history": [
        {"title": "...", "company": "...", "from": "2018-01", "to": "2023-01"}
    ],
    "skills": ["skill1", "skill2"],
    "education": {"degree": "...", "university": "...", "year": 2010},
    "relationships": [
        {"name": "Person Name", "relation": "mentor/colleague/friend"}
    ],
    "personal": {"hobby": "...", "location": "..."}
}

Make it detailed and realistic. Only output JSON."""
        
        response = self.generate(prompt)
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback
        return {
            "name": f"Person_{random.randint(1000, 9999)}",
            "nicknames": [],
            "current_job": {"title": "Engineer", "company": "TechCorp"},
            "job_history": [],
            "skills": [],
            "education": {},
            "relationships": [],
            "personal": {}
        }
    
    def generate_conversation_turn(self, context: str, topic: str) -> str:
        """Generate a realistic conversation turn."""
        prompt = f"""Generate a single realistic Slack/Teams message about: {topic}

Context: {context[:500]}

Requirements:
- Casual workplace tone
- 2-4 sentences
- Include specific details/facts
- May reference other people
- May include updates/changes to previous info

Just output the message, no metadata."""
        
        return self.generate(prompt, temperature=0.9)
    
    def generate_noise_message(self, domain: str) -> str:
        """Generate realistic noise content."""
        prompt = f"""Generate a realistic workplace message about {domain}.

Requirements:
- Sounds like real team communication
- Contains names, projects, dates
- 2-5 sentences
- Similar vocabulary to the domain

Just output the message."""
        
        return self.generate(prompt, temperature=0.9)
    
    def generate_document(self, topic: str, entity_name: str, entity_alias: str) -> str:
        """Generate a document mentioning an entity with a specific alias."""
        prompt = f"""Write a short corporate document (100-200 words) about {topic}.

IMPORTANT: Refer to the person as "{entity_alias}" (not "{entity_name}").

Make it sound like an internal company document, wiki page, or memo.
Include specific details, dates, and other people's names.

Just output the document content."""
        
        return self.generate(prompt, temperature=0.7)


# ============================================================================
# MEGA CHALLENGES
# ============================================================================
@dataclass
class MegaDocument:
    """A document in the mega challenge."""
    id: str
    content: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)
    char_count: int = 0
    
    def __post_init__(self):
        self.char_count = len(self.content)


@dataclass
class MegaQuery:
    """A query for the mega challenge."""
    id: str
    question: str
    answer: str
    reasoning: str
    requires_full_context: bool = True  # Can't be answered with top-k


@dataclass
class MegaChallenge:
    """A mega challenge dataset."""
    name: str
    description: str
    documents: List[MegaDocument]
    queries: List[MegaQuery]
    total_chars: int = 0
    total_tokens_approx: int = 0  # Approx tokens (chars / 4)
    naive_rag_context_limit: int = 32000  # tokens
    
    def __post_init__(self):
        self.total_chars = sum(d.char_count for d in self.documents)
        self.total_tokens_approx = self.total_chars // 4


class ConversationEvolutionChallenge:
    """
    CHALLENGE 1: Massive Conversation with Evolving Facts
    
    Simulates 1000+ turns of workplace conversation.
    Facts CHANGE throughout: jobs, projects, relationships.
    Query asks about CURRENT state - requires finding latest.
    
    Naive RAG FAILS because:
    - Total tokens > context window
    - Top-k retrieval gets random turns, not latest
    - No temporal ordering in retrieval
    """
    
    def __init__(self, llm_generator: LLMGenerator = None):
        self.llm = llm_generator
    
    def generate(self, num_turns: int = 1000, use_llm: bool = True) -> MegaChallenge:
        logger.info(f"Generating {num_turns}-turn conversation evolution challenge...")
        
        # Create the main character with evolving facts
        main_person = "Jordan Taylor"
        aliases = ["Jordan", "J. Taylor", "JT", "Taylor", "Jordan T."]
        
        # Timeline of jobs (each change happens at specific turn)
        job_timeline = [
            (0, "Junior Developer", "StartupXYZ", "$60,000"),
            (150, "Software Engineer", "TechFlow Inc", "$85,000"),
            (350, "Senior Engineer", "TechFlow Inc", "$110,000"),
            (550, "Tech Lead", "MegaCorp", "$140,000"),
            (750, "Engineering Manager", "MegaCorp", "$165,000"),
            (900, "VP of Engineering", "Quantum Dynamics", "$220,000"),
        ]
        
        # Relationship timeline
        relationship_timeline = [
            (0, "single", None),
            (200, "dating", "Alex Morgan"),
            (400, "engaged to", "Alex Morgan"),
            (600, "married to", "Alex Taylor (née Morgan)"),
        ]
        
        # Project timeline
        project_timeline = [
            (0, "Project Alpha"),
            (100, "DataPipeline Initiative"),
            (250, "Cloud Migration"),
            (400, "AI Platform v2"),
            (600, "Enterprise Search"),
            (800, "Project Prometheus"),
            (950, "Quantum Computing Initiative"),
        ]
        
        documents = []
        current_job_idx = 0
        current_rel_idx = 0
        current_proj_idx = 0
        
        base_date = datetime(2019, 1, 1)
        
        for turn in range(num_turns):
            # Update current states based on turn
            while current_job_idx < len(job_timeline) - 1 and job_timeline[current_job_idx + 1][0] <= turn:
                current_job_idx += 1
            while current_rel_idx < len(relationship_timeline) - 1 and relationship_timeline[current_rel_idx + 1][0] <= turn:
                current_rel_idx += 1
            while current_proj_idx < len(project_timeline) - 1 and project_timeline[current_proj_idx + 1][0] <= turn:
                current_proj_idx += 1
            
            _, job_title, company, salary = job_timeline[current_job_idx]
            _, rel_status, partner = relationship_timeline[current_rel_idx]
            _, project = project_timeline[current_proj_idx]
            
            # Use random alias
            alias = random.choice(aliases)
            
            # Generate message content
            if use_llm and self.llm:
                context = f"{alias} works as {job_title} at {company}. Working on {project}."
                if turn % 50 == 0:  # Major update every 50 turns
                    content = self._generate_major_update(alias, job_title, company, project, rel_status, partner)
                else:
                    content = self.llm.generate_conversation_turn(context, "work update")
                    if not content:
                        content = self._generate_fallback_message(alias, job_title, company, project, turn)
            else:
                content = self._generate_fallback_message(alias, job_title, company, project, turn)
            
            # Add noise messages (70% of turns are noise)
            if turn % 10 < 7:
                noise_content = self._generate_noise(turn)
                documents.append(MegaDocument(
                    id=f"noise_{turn}",
                    content=noise_content,
                    timestamp=base_date + timedelta(hours=turn),
                    metadata={"is_noise": True, "turn": turn},
                ))
            
            documents.append(MegaDocument(
                id=f"turn_{turn}",
                content=content,
                timestamp=base_date + timedelta(hours=turn),
                metadata={
                    "turn": turn,
                    "alias_used": alias,
                    "job": job_title,
                    "company": company,
                    "project": project,
                    "is_signal": True,
                },
            ))
        
        # Shuffle to prevent temporal ordering
        random.shuffle(documents)
        
        # Get final states for answers
        final_job = job_timeline[-1]
        final_rel = relationship_timeline[-1]
        final_proj = project_timeline[-1]
        
        queries = [
            MegaQuery(
                id="ce_q1",
                question=f"What is Jordan Taylor's current job title and company?",
                answer=f"{final_job[1]} at {final_job[2]}",
                reasoning=f"Jordan had {len(job_timeline)} different jobs across {num_turns} messages. Naive RAG retrieves random messages mentioning old jobs. Need to find the LATEST message.",
            ),
            MegaQuery(
                id="ce_q2",
                question=f"What is Jordan Taylor's current salary?",
                answer=final_job[3],
                reasoning=f"Salary changed {len(job_timeline)} times. Many messages mention old salaries. Only the latest is correct.",
            ),
            MegaQuery(
                id="ce_q3",
                question=f"What project is Jordan Taylor currently working on?",
                answer=final_proj[1],
                reasoning=f"{len(project_timeline)} different projects mentioned. Must find the most recent one.",
            ),
            MegaQuery(
                id="ce_q4",
                question=f"What is Jordan Taylor's relationship status?",
                answer=f"{final_rel[1]} {final_rel[2]}" if final_rel[2] else final_rel[1],
                reasoning="Relationship evolved from single → dating → engaged → married. Need latest.",
            ),
            MegaQuery(
                id="ce_q5",
                question=f"List all jobs Jordan Taylor has had in chronological order.",
                answer=", ".join([f"{j[1]} at {j[2]}" for j in job_timeline]),
                reasoning=f"Requires finding ALL {len(job_timeline)} job mentions and ordering them. Impossible with top-k.",
            ),
        ]
        
        challenge = MegaChallenge(
            name=f"Conversation Evolution ({num_turns} turns)",
            description=f"Workplace conversation with {num_turns} turns. Facts change {len(job_timeline)} times.",
            documents=documents,
            queries=queries,
        )
        
        logger.info(f"Generated {len(documents)} messages, ~{challenge.total_tokens_approx:,} tokens")
        logger.info(f"Naive RAG context limit: {challenge.naive_rag_context_limit:,} tokens")
        logger.info(f"Data exceeds context by: {challenge.total_tokens_approx / challenge.naive_rag_context_limit:.1f}x")
        
        return challenge
    
    def _generate_major_update(self, alias, job, company, project, rel_status, partner) -> str:
        """Generate a major status update message."""
        templates = [
            f"Big news everyone! {alias} just got promoted to {job} at {company}! 🎉 Currently leading {project}.",
            f"Exciting update: {alias} is now {job} at {company}. The team is thrilled! Working on {project}.",
            f"Announcement: {alias} has accepted the position of {job} at {company}. First project: {project}.",
            f"Career update from {alias}: Started as {job} at {company} this week. Already diving into {project}!",
        ]
        msg = random.choice(templates)
        if partner:
            msg += f" Also, {alias} is {rel_status} {partner}!"
        return msg
    
    def _generate_fallback_message(self, alias, job, company, project, turn) -> str:
        """Generate fallback message without LLM."""
        templates = [
            f"{alias} mentioned the {project} deadline is approaching. The team at {company} is pushing hard.",
            f"Quick update from {alias}: As {job}, the focus this week is on {project} deliverables.",
            f"{alias} from {company} shared progress on {project}. Looking good!",
            f"Standup notes: {alias} ({job}) is working on {project}. No blockers.",
            f"{alias} here. {project} is on track. {company} leadership is pleased.",
            f"FYI from {alias}: {project} sprint review tomorrow. {company} stakeholders attending.",
            f"{alias} posted in #{project.lower().replace(' ', '-')}: making progress as {job}!",
        ]
        return random.choice(templates)
    
    def _generate_noise(self, turn) -> str:
        """Generate noise message."""
        people = ["Sarah", "Mike", "Emma", "Chris", "Lisa", "David", "Amy", "Tom"]
        projects = ["Backend refactor", "UI redesign", "API v3", "Mobile app", "Security audit"]
        topics = [
            f"{random.choice(people)} is OOO tomorrow.",
            f"Anyone want to grab lunch? - {random.choice(people)}",
            f"{random.choice(projects)} status: all tests passing.",
            f"Reminder: Team meeting at 3pm. - {random.choice(people)}",
            f"{random.choice(people)} pushed updates to {random.choice(projects)}.",
            f"Happy Friday everyone! 🎉",
            f"Coffee machine on floor 3 is broken again.",
            f"{random.choice(people)}: Can someone review my PR?",
            f"IT notice: VPN maintenance tonight.",
            f"{random.choice(people)} is looking for feedback on {random.choice(projects)}.",
        ]
        return random.choice(topics)


class EnterpriseKnowledgeBaseChallenge:
    """
    CHALLENGE 2: Massive Enterprise Knowledge Base
    
    5000+ interconnected documents.
    Same entities mentioned across 100+ docs with name variations.
    Complex relationships requiring multi-hop.
    
    Naive RAG FAILS because:
    - 5000 docs, top-10 = 0.2% coverage
    - Answer requires connecting info from 10+ docs
    - Entity variations fool embeddings
    """
    
    def __init__(self, llm_generator: LLMGenerator = None):
        self.llm = llm_generator
    
    def generate(self, num_docs: int = 2000, use_llm: bool = True) -> MegaChallenge:
        logger.info(f"Generating {num_docs}-document enterprise knowledge base...")
        
        # Create company structure
        departments = ["Engineering", "Product", "Sales", "Marketing", "HR", "Finance", "Legal", "Operations"]
        
        # Key people with aliases
        key_people = [
            {
                "name": "Victoria Chen",
                "aliases": ["Victoria", "Vic", "V. Chen", "Dr. Chen", "CEO Chen", "The Chief"],
                "role": "CEO",
                "reports": [],
                "facts": [
                    "Founded the company in 2015",
                    "Has PhD from Stanford",
                    "Previously VP at Google",
                    "Led the Series D funding of $500M",
                    "Lives in San Francisco",
                ],
            },
            {
                "name": "Marcus Williams",
                "aliases": ["Marcus", "Marc", "M. Williams", "CTO Marcus", "Tech Lead Williams"],
                "role": "CTO",
                "reports": ["Victoria Chen"],
                "facts": [
                    "Joined in 2016 as first engineer",
                    "Architected the core platform",
                    "Has 50 patents",
                    "MIT graduate",
                    "Expert in distributed systems",
                ],
            },
            {
                "name": "Elena Rodriguez",
                "aliases": ["Elena", "E. Rodriguez", "CFO Rodriguez", "Finance Lead"],
                "role": "CFO",
                "reports": ["Victoria Chen"],
                "facts": [
                    "Joined from Goldman Sachs",
                    "Led the IPO preparation",
                    "Manages $2B budget",
                    "Harvard MBA",
                ],
            },
        ]
        
        # Create interconnected documents
        documents = []
        doc_id = 0
        
        # Create multiple docs per person with different aliases
        for person in key_people:
            for alias in person["aliases"]:
                for fact in person["facts"]:
                    if use_llm and self.llm:
                        content = self.llm.generate_document(
                            topic=f"{person['role']} activities and {fact}",
                            entity_name=person["name"],
                            entity_alias=alias
                        )
                        if not content:
                            content = self._fallback_doc(alias, person["role"], fact)
                    else:
                        content = self._fallback_doc(alias, person["role"], fact)
                    
                    documents.append(MegaDocument(
                        id=f"person_doc_{doc_id}",
                        content=content,
                        timestamp=datetime.now() - timedelta(days=random.randint(1, 365)),
                        metadata={"person": person["name"], "alias": alias, "fact": fact},
                    ))
                    doc_id += 1
        
        # Create relationship documents (A reports to B)
        for person in key_people:
            for report_to in person["reports"]:
                report_person = next((p for p in key_people if p["name"] == report_to), None)
                if report_person:
                    alias1 = random.choice(person["aliases"])
                    alias2 = random.choice(report_person["aliases"])
                    content = f"""
                    Organizational Update
                    
                    {alias1}, serving as {person['role']}, reports directly to {alias2}.
                    This reporting structure was established to streamline decision-making.
                    {alias1} and {alias2} meet weekly to discuss strategic initiatives.
                    """
                    documents.append(MegaDocument(
                        id=f"rel_doc_{doc_id}",
                        content=content.strip(),
                        timestamp=datetime.now() - timedelta(days=random.randint(1, 100)),
                        metadata={"relationship": f"{person['name']} -> {report_to}"},
                    ))
                    doc_id += 1
        
        # Generate MASSIVE noise (department docs, project docs, meeting notes)
        noise_templates = [
            "Team meeting notes from {dept} department. Discussed Q{q} priorities.",
            "{dept} quarterly review. Performance metrics shared.",
            "Project update: {project} milestone achieved.",
            "HR announcement: New benefits program for {dept}.",
            "Training session: {topic} fundamentals.",
            "{dept} team outing scheduled for next month.",
            "Budget review for {dept} department completed.",
            "New hire onboarding in {dept} department.",
            "Process improvement initiative in {dept}.",
            "Cross-functional meeting: {dept} and Engineering sync.",
        ]
        
        projects = ["Phoenix", "Titan", "Apollo", "Mercury", "Neptune", "Atlas", "Orion"]
        topics = ["Cloud", "AI/ML", "Security", "Data", "DevOps", "Mobile", "Analytics"]
        
        while len(documents) < num_docs:
            template = random.choice(noise_templates)
            content = template.format(
                dept=random.choice(departments),
                q=random.randint(1, 4),
                project=random.choice(projects),
                topic=random.choice(topics),
            )
            # Add some realistic padding
            content += f" Reference: DOC-{doc_id}. Last updated: {datetime.now().date()}."
            
            documents.append(MegaDocument(
                id=f"noise_{doc_id}",
                content=content,
                timestamp=datetime.now() - timedelta(days=random.randint(1, 365)),
                metadata={"is_noise": True},
            ))
            doc_id += 1
        
        random.shuffle(documents)
        
        queries = [
            MegaQuery(
                id="ekb_q1",
                question="Who founded the company and what is their educational background?",
                answer="Victoria Chen founded the company in 2015. She has a PhD from Stanford.",
                reasoning=f"Victoria mentioned with 6 different aliases across {len(documents)} docs. Naive RAG search for 'founder' retrieves random docs.",
            ),
            MegaQuery(
                id="ekb_q2",
                question="Who does the CTO report to?",
                answer="Marcus Williams (CTO) reports to Victoria Chen (CEO).",
                reasoning="Relationship mentioned using aliases like 'Marc' and 'V. Chen'. Entity resolution needed.",
            ),
            MegaQuery(
                id="ekb_q3",
                question="What is Victoria Chen's previous work experience?",
                answer="Previously VP at Google",
                reasoning="This fact appears in docs using alias 'The Chief' or 'Dr. Chen'. Name mismatch with query.",
            ),
            MegaQuery(
                id="ekb_q4",
                question="How much funding did the company raise and who led it?",
                answer="$500M Series D, led by Victoria Chen",
                reasoning="Funding mentioned in doc using 'CEO Chen'. Query uses 'Victoria Chen'.",
            ),
            MegaQuery(
                id="ekb_q5",
                question="List all executives and their backgrounds.",
                answer="Victoria Chen (CEO) - Stanford PhD, ex-Google VP. Marcus Williams (CTO) - MIT, 50 patents. Elena Rodriguez (CFO) - Harvard MBA, ex-Goldman Sachs.",
                reasoning=f"Requires finding docs for 3 people across {len(documents)} docs, each with multiple aliases. Impossible with top-k.",
            ),
        ]
        
        challenge = MegaChallenge(
            name=f"Enterprise Knowledge Base ({num_docs} docs)",
            description=f"Corporate wiki with {num_docs} interconnected documents and entity variations",
            documents=documents,
            queries=queries,
        )
        
        logger.info(f"Generated {len(documents)} documents, ~{challenge.total_tokens_approx:,} tokens")
        logger.info(f"Key people: {len(key_people)}, each with {len(key_people[0]['aliases'])} aliases")
        
        return challenge
    
    def _fallback_doc(self, alias: str, role: str, fact: str) -> str:
        """Generate fallback document."""
        templates = [
            f"According to internal records, {alias} ({role}) {fact}. This information is verified.",
            f"Profile Update: {alias}, serving as {role}, {fact}. Last reviewed this quarter.",
            f"Executive Summary: {alias} ({role}) - Key fact: {fact}.",
            f"Leadership Wiki: {alias} | Role: {role} | Notable: {fact}",
        ]
        return random.choice(templates)


# ============================================================================
# EVALUATOR
# ============================================================================
class MegaEvaluator:
    """Evaluate GraphMem vs Naive RAG on mega challenges."""
    
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
        self._llm_provider = None
    
    def _get_llm_provider(self):
        """Get LLM provider for generation."""
        if self._llm_provider:
            return self._llm_provider
        
        from graphmem.llm.providers import get_llm_provider
        
        provider_name = "azure_openai" if self.provider == "azure" else self.provider
        self._llm_provider = get_llm_provider(
            provider=provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.llm_model,
            api_version="2024-08-01-preview",
            deployment=self.azure_deployment,
        )
        return self._llm_provider
    
    def _init_graphmem(self, db_path: str = "mega_bench.db"):
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
    
    def _init_naive_rag(self, context_limit: int = 32000):
        """Initialize naive RAG with context limit."""
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
        
        class NaiveRAGWithLimit:
            def __init__(self, llm, embedder, token_limit):
                self.llm = llm
                self.embedder = embedder
                self.token_limit = token_limit
                self.chunks = []
                self.embeddings = []
            
            def ingest(self, docs):
                logger.info(f"   NaiveRAG: Ingesting {len(docs)} documents...")
                for i, doc in enumerate(docs):
                    try:
                        emb = self.embedder.embed_text(doc.content)
                        self.chunks.append(doc.content)
                        self.embeddings.append(emb)
                    except Exception as e:
                        if "rate" in str(e).lower():
                            time.sleep(2)
                            try:
                                emb = self.embedder.embed_text(doc.content)
                                self.chunks.append(doc.content)
                                self.embeddings.append(emb)
                            except:
                                pass
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"   NaiveRAG: {i+1}/{len(docs)} embedded")
            
            def query(self, question, top_k=10):
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
                
                # Build context UP TO token limit
                context_chunks = []
                total_tokens = 0
                for i, _ in sims[:top_k * 2]:  # Get more, trim to fit
                    chunk = self.chunks[i]
                    chunk_tokens = len(chunk) // 4
                    if total_tokens + chunk_tokens > self.token_limit - 1000:  # Leave room for prompt
                        break
                    context_chunks.append(chunk)
                    total_tokens += chunk_tokens
                
                context = "\n\n---\n\n".join(context_chunks)
                
                prompt = f"""Answer based ONLY on the provided context.
If the answer is not in the context, say "Cannot determine from available context."

Context:
{context}

Question: {question}

Answer:"""
                
                return self.llm.chat([{"role": "user", "content": prompt}])
        
        return NaiveRAGWithLimit(llm, embedder, context_limit)
    
    def evaluate(self, challenge: MegaChallenge, use_llm_generation: bool = True) -> Dict:
        """Evaluate both systems."""
        print(f"\n{'='*80}")
        print(f"🧪 MEGA CHALLENGE: {challenge.name}")
        print(f"{'='*80}")
        print(f"   Documents: {len(challenge.documents):,}")
        print(f"   Total tokens (approx): {challenge.total_tokens_approx:,}")
        print(f"   Naive RAG context limit: {challenge.naive_rag_context_limit:,}")
        print(f"   Data exceeds context by: {challenge.total_tokens_approx / challenge.naive_rag_context_limit:.1f}x")
        print(f"\n   🚨 This means Naive RAG CANNOT see all the data!")
        
        # Initialize
        gm = self._init_graphmem()
        rag = self._init_naive_rag(challenge.naive_rag_context_limit)
        
        # Ingest
        print(f"\n📥 Ingesting...")
        
        docs_for_gm = [{"id": d.id, "content": d.content} for d in challenge.documents]
        
        print("   GraphMem (20 workers, with entity resolution)...")
        start = time.time()
        gm.ingest_batch(documents=docs_for_gm, max_workers=20, aggressive=True, show_progress=True)
        gm.evolve()
        gm_time = time.time() - start
        print(f"   GraphMem ingestion: {gm_time:.1f}s")
        
        print("   Naive RAG (embedding only)...")
        start = time.time()
        rag.ingest(challenge.documents)
        rag_time = time.time() - start
        print(f"   Naive RAG ingestion: {rag_time:.1f}s")
        
        # Query
        print(f"\n🔍 Running {len(challenge.queries)} queries...")
        
        gm_correct = 0
        rag_correct = 0
        
        for q in challenge.queries:
            print(f"\n   Q: {q.question[:70]}...")
            print(f"   Expected: {q.answer[:70]}...")
            print(f"   Why hard: {q.reasoning[:80]}...")
            
            # GraphMem
            try:
                gm_resp = gm.query(q.question)
                gm_answer = gm_resp.answer
            except Exception as e:
                gm_answer = f"Error: {e}"
            
            # Naive RAG
            try:
                rag_answer = rag.query(q.question)
            except Exception as e:
                rag_answer = f"Error: {e}"
            
            # Check (simple contains)
            gm_ok = any(part.lower() in gm_answer.lower() for part in q.answer.split(",")[0].split())
            rag_ok = any(part.lower() in rag_answer.lower() for part in q.answer.split(",")[0].split())
            
            if gm_ok:
                gm_correct += 1
            if rag_ok:
                rag_correct += 1
            
            print(f"   GraphMem: {'✅' if gm_ok else '❌'} {gm_answer[:60]}...")
            print(f"   NaiveRAG: {'✅' if rag_ok else '❌'} {rag_answer[:60]}...")
        
        # Results
        print(f"\n{'='*80}")
        print("📊 RESULTS")
        print(f"{'='*80}")
        print(f"   GraphMem: {gm_correct}/{len(challenge.queries)} ({100*gm_correct/len(challenge.queries):.0f}%)")
        print(f"   NaiveRAG: {rag_correct}/{len(challenge.queries)} ({100*rag_correct/len(challenge.queries):.0f}%)")
        
        if gm_correct > rag_correct:
            margin = gm_correct - rag_correct
            print(f"\n   🏆 GraphMem WINS by {margin} questions!")
            print(f"   ✅ GraphMem handles data that EXCEEDS context window")
        else:
            print(f"\n   ⚠️ Unexpected result - need larger dataset or more complex queries")
        
        return {
            "challenge": challenge.name,
            "total_docs": len(challenge.documents),
            "total_tokens": challenge.total_tokens_approx,
            "context_limit": challenge.naive_rag_context_limit,
            "exceeds_by": f"{challenge.total_tokens_approx / challenge.naive_rag_context_limit:.1f}x",
            "graphmem_correct": gm_correct,
            "naive_rag_correct": rag_correct,
            "total_queries": len(challenge.queries),
        }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Mega Synthetic Benchmark")
    
    parser.add_argument("--scenario", default="conversation",
                        choices=["conversation", "enterprise", "all"])
    parser.add_argument("--num-turns", type=int, default=500, help="Turns for conversation")
    parser.add_argument("--num-docs", type=int, default=1000, help="Docs for enterprise")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for generation")
    
    parser.add_argument("--provider", default="azure")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--azure-deployment", default="gpt-4.1-mini")
    parser.add_argument("--azure-embedding-deployment", default="text-embedding-3-small")
    
    parser.add_argument("--output", default="mega_results.json")
    
    args = parser.parse_args()
    
    evaluator = MegaEvaluator(
        provider=args.provider,
        api_key=args.api_key,
        api_base=args.azure_endpoint,
        llm_model=args.azure_deployment,
        embedding_model=args.azure_embedding_deployment,
        azure_deployment=args.azure_deployment,
        azure_embedding_deployment=args.azure_embedding_deployment,
    )
    
    llm_gen = None
    if args.use_llm:
        llm_gen = LLMGenerator(evaluator._get_llm_provider())
    
    results = []
    
    if args.scenario in ["conversation", "all"]:
        gen = ConversationEvolutionChallenge(llm_gen)
        challenge = gen.generate(args.num_turns, use_llm=args.use_llm)
        result = evaluator.evaluate(challenge)
        results.append(result)
    
    if args.scenario in ["enterprise", "all"]:
        gen = EnterpriseKnowledgeBaseChallenge(llm_gen)
        challenge = gen.generate(args.num_docs, use_llm=args.use_llm)
        result = evaluator.evaluate(challenge)
        results.append(result)
    
    # Save
    with open(args.output, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()


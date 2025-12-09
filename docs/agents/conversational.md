# Conversational Agents

Build chatbots and conversational AI with persistent memory.

## Pattern Overview

A conversational agent:
- Remembers past conversations
- Learns user preferences
- Maintains context across sessions
- Forgets irrelevant details over time

---

## Basic Implementation

```python
from graphmem import GraphMem, MemoryConfig, MemoryImportance

class ConversationalAgent:
    """Agent that remembers conversations across sessions."""
    
    def __init__(self, user_id: str):
        self.config = MemoryConfig(
            llm_provider="openai",
            llm_api_key="sk-...",
            llm_model="gpt-4o-mini",
            embedding_provider="openai",
            embedding_api_key="sk-...",
            embedding_model="text-embedding-3-small",
            
            # Persist to SQLite
            turso_db_path=f"memories/{user_id}.db",
            
            # User isolation
            user_id=user_id,
            
            # Evolution settings
            evolution_enabled=True,
            decay_enabled=True,
            decay_half_life_days=30,
        )
        self.memory = GraphMem(self.config, memory_id=f"agent_{user_id}", user_id=user_id)
        self.user_id = user_id
    
    def chat(self, user_message: str) -> str:
        """Process user message and generate response."""
        
        # 1. Store the user's message
        self.memory.ingest(
            f"User said: {user_message}",
            metadata={"type": "user_message"},
            importance=MemoryImportance.MEDIUM,
        )
        
        # 2. Query memory for relevant context
        response = self.memory.query(user_message)
        
        # 3. Generate response using your LLM
        agent_response = self._generate_response(
            user_message, 
            response.context
        )
        
        # 4. Store the agent's response
        self.memory.ingest(
            f"Agent responded: {agent_response}",
            metadata={"type": "agent_response"},
            importance=MemoryImportance.LOW,
        )
        
        return agent_response
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with memory context."""
        # Your LLM call here
        prompt = f"""
        You are a helpful assistant with memory of past conversations.
        
        Relevant context from memory:
        {context}
        
        User message: {query}
        
        Respond helpfully using the context when relevant.
        """
        # Call your LLM...
        pass
    
    def end_session(self):
        """Consolidate memories at end of session."""
        self.memory.evolve()
```

---

## Usage Example

```python
# Create agent for user
agent = ConversationalAgent(user_id="user_123")

# First session
agent.chat("My name is Alice and I work at Google.")
agent.chat("I'm interested in machine learning.")
agent.end_session()

# Later session (memories persist!)
agent.chat("What do you know about me?")
# → "You're Alice, you work at Google, and you're interested in machine learning."
```

---

## Memory Priority

Prioritize what to remember:

```python
def chat(self, user_message: str) -> str:
    # Detect importance based on content
    importance = self._classify_importance(user_message)
    
    self.memory.ingest(
        f"User said: {user_message}",
        importance=importance,
    )
    # ...

def _classify_importance(self, message: str) -> MemoryImportance:
    """Classify message importance."""
    # Personal facts (name, preferences) - high importance
    if any(kw in message.lower() for kw in ["my name is", "i am", "i work"]):
        return MemoryImportance.HIGH
    
    # Preferences - medium
    if any(kw in message.lower() for kw in ["i like", "i prefer", "favorite"]):
        return MemoryImportance.MEDIUM
    
    # Chitchat - low (will decay)
    return MemoryImportance.LOW
```

---

## Session Management

### Per-Session Memory

```python
class SessionAwareAgent:
    def __init__(self, user_id: str, session_id: str):
        self.memory = GraphMem(
            config,
            user_id=user_id,
            memory_id=session_id,  # Isolate by session
        )
```

### Cross-Session Memory

```python
class CrossSessionAgent:
    def __init__(self, user_id: str):
        # General memory (persists across sessions)
        self.long_term = GraphMem(
            config,
            user_id=user_id,
            memory_id="long_term",
        )
        # Session-specific memory
        self.session = GraphMem(
            config,
            user_id=user_id,
            memory_id=f"session_{datetime.now().isoformat()}",
        )
    
    def chat(self, message: str) -> str:
        # Store in both
        self.session.ingest(message)
        
        # Important facts go to long-term
        if self._is_important(message):
            self.long_term.ingest(message)
        
        # Query both
        session_context = self.session.query(message).context
        long_term_context = self.long_term.query(message).context
        
        return self._generate_response(
            message,
            session_context + long_term_context
        )
```

---

## FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

# Store agent instances
agents = {}

def get_agent(user_id: str) -> ConversationalAgent:
    if user_id not in agents:
        agents[user_id] = ConversationalAgent(user_id)
    return agents[user_id]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float

@app.post("/chat/{user_id}", response_model=ChatResponse)
async def chat(user_id: str, request: ChatRequest):
    agent = get_agent(user_id)
    response = agent.chat(request.message)
    return ChatResponse(response=response, confidence=0.9)

@app.post("/end-session/{user_id}")
async def end_session(user_id: str):
    agent = get_agent(user_id)
    agent.end_session()
    return {"status": "session ended"}
```

---

## Best Practices

1. **Set appropriate decay** - Casual chat should decay, facts shouldn't
2. **End sessions properly** - Call `evolve()` to consolidate
3. **Classify importance** - Not all messages are equal
4. **Use session IDs** - Isolate conversations when needed
5. **Handle errors gracefully** - Fallback when memory fails


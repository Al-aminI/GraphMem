#!/usr/bin/env python3
"""
GraphMem OpenAI-Compatible API Server Example

This example demonstrates how to:
1. Start the GraphMem OpenAI-compatible server
2. Use it with the OpenAI Python client
3. Use it with curl commands
4. Benchmark memory-augmented responses

Prerequisites:
    pip install graphmem openai httpx

Usage:
    # Start the server
    python -m graphmem.openai_server.cli --port 8000

    # Then run this example
    python examples/openai_server_example.py
"""

import os
import time
import asyncio
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================

# Server URL (change if running on different host/port)
SERVER_URL = os.getenv("GRAPHMEM_SERVER_URL", "http://localhost:8000/v1")

# Initialize OpenAI client pointing to our server
client = OpenAI(
    base_url=SERVER_URL,
    api_key="not-needed"  # API key not required for local server
)


# ============================================================================
# Example Functions
# ============================================================================

def test_health():
    """Test the health endpoint."""
    import httpx
    
    base_url = SERVER_URL.replace("/v1", "")
    response = httpx.get(f"{base_url}/health")
    print("Health Check:")
    print(response.json())
    print()


def list_models():
    """List available models."""
    models = client.models.list()
    print("Available Models:")
    for model in models.data:
        print(f"  - {model.id} (owned by: {model.owned_by})")
    print()


def chat_completion_basic():
    """Basic chat completion example."""
    print("=" * 60)
    print("Basic Chat Completion")
    print("=" * 60)
    
    start = time.time()
    
    response = client.chat.completions.create(
        model="graphmem-turso",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to memory."},
            {"role": "user", "content": "What do you know about me or any previous conversations?"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    elapsed = time.time() - start
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Usage: {response.usage}")
    print(f"Time: {elapsed:.2f}s")
    print()


def chat_completion_with_memory_query():
    """Chat completion with a specific memory query."""
    print("=" * 60)
    print("Memory-Augmented Chat Completion")
    print("=" * 60)
    
    # Query something that might be in memory
    response = client.chat.completions.create(
        model="graphmem-turso",
        messages=[
            {"role": "user", "content": "Who is the CEO of Tesla?"}
        ],
        max_tokens=150
    )
    
    print(f"Query: Who is the CEO of Tesla?")
    print(f"Response: {response.choices[0].message.content}")
    print()


def chat_completion_streaming():
    """Streaming chat completion example."""
    print("=" * 60)
    print("Streaming Chat Completion")
    print("=" * 60)
    
    print("Response: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="graphmem-turso",
        messages=[
            {"role": "user", "content": "Tell me a short story about AI and memory."}
        ],
        max_tokens=200,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print("\n")


def embeddings_example():
    """Generate embeddings example."""
    print("=" * 60)
    print("Embeddings")
    print("=" * 60)
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=["Hello, world!", "GraphMem is amazing!"]
    )
    
    print(f"Generated {len(response.data)} embeddings")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
    print(f"Usage: {response.usage}")
    print()


def benchmark_latency(num_requests: int = 10):
    """Benchmark response latency."""
    print("=" * 60)
    print(f"Latency Benchmark ({num_requests} requests)")
    print("=" * 60)
    
    latencies = []
    
    for i in range(num_requests):
        start = time.time()
        
        response = client.chat.completions.create(
            model="graphmem-turso",
            messages=[
                {"role": "user", "content": f"What is {i + 1} + {i + 1}?"}
            ],
            max_tokens=50
        )
        
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"  Request {i + 1}: {elapsed:.3f}s")
    
    avg = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print(f"\nResults:")
    print(f"  Average: {avg:.3f}s")
    print(f"  Min: {min_lat:.3f}s")
    print(f"  Max: {max_lat:.3f}s")
    print()


def curl_examples():
    """Print curl command examples."""
    print("=" * 60)
    print("cURL Examples")
    print("=" * 60)
    
    base_url = SERVER_URL.replace("/v1", "")
    
    print(f"""
# Health check
curl {base_url}/health

# List models
curl {SERVER_URL}/models

# Chat completion
curl -X POST {SERVER_URL}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "graphmem-turso",
    "messages": [
      {{"role": "user", "content": "Hello, what do you know about me?"}}
    ],
    "max_tokens": 100
  }}'

# Streaming chat completion
curl -X POST {SERVER_URL}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "graphmem-turso",
    "messages": [
      {{"role": "user", "content": "Tell me a story"}}
    ],
    "max_tokens": 200,
    "stream": true
  }}'

# Query memory directly
curl -X POST "{base_url}/v1/memory/query?query=What%20is%20GraphMem"

# Get memory stats
curl {base_url}/v1/memory/stats
    """)


def langchain_example():
    """Example using LangChain with GraphMem server."""
    print("=" * 60)
    print("LangChain Integration Example")
    print("=" * 60)
    
    print("""
# Using LangChain with GraphMem OpenAI-Compatible Server

from langchain_openai import ChatOpenAI

# Point LangChain to your GraphMem server
llm = ChatOpenAI(
    model="graphmem-turso",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="not-needed",
    temperature=0.7
)

# Use it like any other LLM
response = llm.invoke("What do you remember about our previous conversations?")
print(response.content)

# Or in a chain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

chain = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)

chain.run("Hello, I'm Alice. Remember my name.")
chain.run("What's my name?")
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║      GraphMem OpenAI-Compatible API - Examples                 ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Test health first
        test_health()
        
        # List models
        list_models()
        
        # Basic chat completion
        chat_completion_basic()
        
        # Memory-augmented query
        chat_completion_with_memory_query()
        
        # Streaming
        chat_completion_streaming()
        
        # Embeddings
        try:
            embeddings_example()
        except Exception as e:
            print(f"Embeddings skipped (may not be configured): {e}\n")
        
        # Benchmark
        benchmark_latency(5)
        
        # Print curl examples
        curl_examples()
        
        # Print LangChain example
        langchain_example()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m graphmem.openai_server.cli --port 8000")


if __name__ == "__main__":
    main()


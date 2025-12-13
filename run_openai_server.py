#!/usr/bin/env python3
"""
GraphMem OpenAI-Compatible Server - Quick Start

This script starts the GraphMem OpenAI-compatible server with your configuration.
Modify the CONFIG dictionary below with your credentials.

Usage:
    python run_openai_server.py

Once running, use any OpenAI-compatible client:
    
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    
    response = client.chat.completions.create(
        model="graphmem-turso",
        messages=[{"role": "user", "content": "What do you know about me?"}]
    )
    print(response.choices[0].message.content)
"""

import os
import sys

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

CONFIG = {
    # LLM Configuration
    "llm_provider": os.getenv("GRAPHMEM_LLM_PROVIDER", "azure_openai"),  # azure_openai, openai, openrouter, ollama
    "llm_api_key": os.getenv("GRAPHMEM_LLM_API_KEY", "YOUR_API_KEY"),
    "llm_api_base": os.getenv("GRAPHMEM_LLM_API_BASE", "https://your-endpoint.openai.azure.com/"),
    "llm_model": os.getenv("GRAPHMEM_LLM_MODEL", "gpt-4.1-mini"),
    
    # Embedding Configuration
    "embedding_provider": os.getenv("GRAPHMEM_EMBEDDING_PROVIDER", "azure_openai"),
    "embedding_api_key": os.getenv("GRAPHMEM_EMBEDDING_API_KEY", "YOUR_API_KEY"),
    "embedding_api_base": os.getenv("GRAPHMEM_EMBEDDING_API_BASE", "https://your-endpoint.openai.azure.com/"),
    "embedding_model": os.getenv("GRAPHMEM_EMBEDDING_MODEL", "text-embedding-3-small"),
    
    # Turso Configuration (your pre-ingested database)
    "turso_db_path": os.getenv("GRAPHMEM_TURSO_DB_PATH", "graphmem_local.db"),  # Local SQLite file
    "turso_url": os.getenv("GRAPHMEM_TURSO_URL", ""),  # Cloud sync (optional)
    "turso_auth_token": os.getenv("GRAPHMEM_TURSO_AUTH_TOKEN", ""),  # Auth token (optional)
    
    # Memory Configuration (must match your ingested data)
    "memory_id": os.getenv("GRAPHMEM_MEMORY_ID", "benchmark-memory"),  # Change to match your ingested memory_id
    "user_id": os.getenv("GRAPHMEM_USER_ID", "benchmark-user"),      # Change to match your ingested user_id
    
    # Model name reported in API responses
    "model_name": os.getenv("GRAPHMEM_MODEL_NAME", "graphmem-turso"),
}

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║      GraphMem OpenAI-Compatible API Server                     ║
║      Memory-Augmented LLM Inference                            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Starting server on http://{HOST}:{PORT}")
    print(f"📚 Model: {CONFIG['model_name']}")
    print(f"🧠 Memory ID: {CONFIG['memory_id']}")
    print(f"👤 User ID: {CONFIG['user_id']}")
    print(f"💾 Turso DB: {CONFIG['turso_db_path']}")
    print()
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print()
    print("Example usage with OpenAI client:")
    print('  client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")')
    print('  response = client.chat.completions.create(')
    print('      model="graphmem-turso",')
    print('      messages=[{"role": "user", "content": "Hello!"}]')
    print('  )')
    print()
    
    # Import and run
    from graphmem.openai_server.server import run_server
    
    run_server(
        host=HOST,
        port=PORT,
        config=CONFIG,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()


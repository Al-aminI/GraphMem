#!/usr/bin/env python3
"""
GraphMem OpenAI-Compatible API Server CLI

Usage:
    # Using environment variables
    export GRAPHMEM_LLM_API_KEY="your-key"
    export GRAPHMEM_LLM_API_BASE="https://your-endpoint"
    export GRAPHMEM_TURSO_URL="libsql://your-db.turso.io"
    export GRAPHMEM_TURSO_AUTH_TOKEN="your-token"
    python -m graphmem.openai_server.cli --port 8000

    # Or with hardcoded config
    python -m graphmem.openai_server.cli --config config.json
"""

import argparse
import json
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="GraphMem OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with environment variables
  python -m graphmem.openai_server.cli --port 8000

  # Start server with config file
  python -m graphmem.openai_server.cli --config config.json

  # Start server with explicit options
  python -m graphmem.openai_server.cli \\
    --llm-provider azure_openai \\
    --llm-model gpt-4.1-mini \\
    --turso-db-path graphmem_local.db \\
    --memory-id my-memory \\
    --user-id my-user

Environment Variables:
  GRAPHMEM_LLM_PROVIDER      LLM provider (azure_openai, openai, openrouter, ollama)
  GRAPHMEM_LLM_API_KEY       LLM API key
  GRAPHMEM_LLM_API_BASE      LLM API base URL
  GRAPHMEM_LLM_MODEL         LLM model name
  GRAPHMEM_EMBEDDING_PROVIDER Embedding provider
  GRAPHMEM_EMBEDDING_API_KEY Embedding API key
  GRAPHMEM_EMBEDDING_API_BASE Embedding API base URL
  GRAPHMEM_EMBEDDING_MODEL   Embedding model name
  GRAPHMEM_TURSO_DB_PATH     Local Turso database path
  GRAPHMEM_TURSO_URL         Turso cloud URL (optional)
  GRAPHMEM_TURSO_AUTH_TOKEN  Turso auth token (optional)
  GRAPHMEM_MEMORY_ID         Memory ID for persistence
  GRAPHMEM_USER_ID           User ID for persistence
  GRAPHMEM_MODEL_NAME        Model name to report in API
        """
    )
    
    # Server options
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    
    # LLM options
    parser.add_argument("--llm-provider", help="LLM provider (azure_openai, openai, openrouter, ollama)")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument("--llm-api-base", help="LLM API base URL")
    parser.add_argument("--llm-model", help="LLM model name")
    
    # Embedding options
    parser.add_argument("--embedding-provider", help="Embedding provider")
    parser.add_argument("--embedding-api-key", help="Embedding API key")
    parser.add_argument("--embedding-api-base", help="Embedding API base URL")
    parser.add_argument("--embedding-model", help="Embedding model name")
    
    # Turso options
    parser.add_argument("--turso-db-path", help="Local Turso database path")
    parser.add_argument("--turso-url", help="Turso cloud URL")
    parser.add_argument("--turso-auth-token", help="Turso auth token")
    
    # Memory options
    parser.add_argument("--memory-id", help="Memory ID for persistence")
    parser.add_argument("--user-id", help="User ID for persistence")
    parser.add_argument("--model-name", help="Model name to report in API responses")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {}
    
    # Load from config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)
    
    # Override with command-line arguments
    arg_mapping = {
        'llm_provider': args.llm_provider,
        'llm_api_key': args.llm_api_key,
        'llm_api_base': args.llm_api_base,
        'llm_model': args.llm_model,
        'embedding_provider': args.embedding_provider,
        'embedding_api_key': args.embedding_api_key,
        'embedding_api_base': args.embedding_api_base,
        'embedding_model': args.embedding_model,
        'turso_db_path': args.turso_db_path,
        'turso_url': args.turso_url,
        'turso_auth_token': args.turso_auth_token,
        'memory_id': args.memory_id,
        'user_id': args.user_id,
        'model_name': args.model_name,
    }
    
    for key, value in arg_mapping.items():
        if value is not None:
            config[key] = value
    
    # Print startup info
    print("""
╔════════════════════════════════════════════════════════════════╗
║      GraphMem OpenAI-Compatible API Server                     ║
║      Memory-Augmented LLM Inference                            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Model: {config.get('model_name', os.getenv('GRAPHMEM_MODEL_NAME', 'graphmem-turso'))}")
    logger.info(f"Memory ID: {config.get('memory_id', os.getenv('GRAPHMEM_MEMORY_ID', 'benchmark-memory'))}")
    logger.info(f"Turso DB: {config.get('turso_db_path', os.getenv('GRAPHMEM_TURSO_DB_PATH', 'graphmem_local.db'))}")
    
    # Import and run server
    from .server import run_server
    
    run_server(
        host=args.host,
        port=args.port,
        config=config if config else None,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()


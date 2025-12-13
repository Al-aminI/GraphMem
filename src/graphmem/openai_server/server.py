"""
GraphMem OpenAI-Compatible API Server

FastAPI server implementation that provides OpenAI-compatible endpoints
for GraphMem's memory-augmented responses.
"""

import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsResponse,
    ErrorResponse,
    ErrorDetail,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from .serving_engine import GraphMemServingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration - Hardcoded for benchmarking
# ============================================================================

# These will be loaded from environment or hardcoded
DEFAULT_CONFIG = {
    # LLM Configuration
    "llm_provider": os.getenv("GRAPHMEM_LLM_PROVIDER", "azure_openai"),
    "llm_api_key": os.getenv("GRAPHMEM_LLM_API_KEY", ""),
    "llm_api_base": os.getenv("GRAPHMEM_LLM_API_BASE", ""),
    "llm_model": os.getenv("GRAPHMEM_LLM_MODEL", "gpt-4.1-mini"),
    
    # Embedding Configuration  
    "embedding_provider": os.getenv("GRAPHMEM_EMBEDDING_PROVIDER", "azure_openai"),
    "embedding_api_key": os.getenv("GRAPHMEM_EMBEDDING_API_KEY", ""),
    "embedding_api_base": os.getenv("GRAPHMEM_EMBEDDING_API_BASE", ""),
    "embedding_model": os.getenv("GRAPHMEM_EMBEDDING_MODEL", "text-embedding-3-small"),
    
    # Turso Configuration
    "turso_db_path": os.getenv("GRAPHMEM_TURSO_DB_PATH", "graphmem_local.db"),
    "turso_url": os.getenv("GRAPHMEM_TURSO_URL", ""),
    "turso_auth_token": os.getenv("GRAPHMEM_TURSO_AUTH_TOKEN", ""),
    
    # Memory Configuration
    "memory_id": os.getenv("GRAPHMEM_MEMORY_ID", "benchmark-memory"),
    "user_id": os.getenv("GRAPHMEM_USER_ID", "benchmark-user"),
    
    # Model name to report
    "model_name": os.getenv("GRAPHMEM_MODEL_NAME", "graphmem-turso"),
}


# Global instances
_graphmem_instance = None
_serving_engine = None


def get_serving_engine() -> GraphMemServingEngine:
    """Get the global serving engine instance."""
    global _serving_engine
    if _serving_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _serving_engine


async def initialize_graphmem(config: Dict[str, Any] = None):
    """Initialize GraphMem with the given configuration."""
    global _graphmem_instance, _serving_engine
    
    config = config or DEFAULT_CONFIG
    
    try:
        from ..core.memory import GraphMem, MemoryConfig
        
        # Create memory config
        memory_config = MemoryConfig(
            # LLM settings
            llm_provider=config["llm_provider"],
            llm_api_key=config["llm_api_key"],
            llm_api_base=config["llm_api_base"],
            llm_model=config["llm_model"],
            
            # Embedding settings
            embedding_provider=config["embedding_provider"],
            embedding_api_key=config["embedding_api_key"],
            embedding_api_base=config["embedding_api_base"],
            embedding_model=config["embedding_model"],
            
            # Turso settings
            turso_db_path=config["turso_db_path"],
            turso_url=config["turso_url"] if config["turso_url"] else None,
            turso_auth_token=config["turso_auth_token"] if config["turso_auth_token"] else None,
        )
        
        # Initialize GraphMem
        _graphmem_instance = GraphMem(
            config=memory_config,
            memory_id=config["memory_id"],
            user_id=config["user_id"]
        )
        
        # Initialize serving engine
        _serving_engine = GraphMemServingEngine(
            graphmem_instance=_graphmem_instance,
            model_name=config["model_name"]
        )
        
        logger.info(f"GraphMem initialized successfully with model: {config['model_name']}")
        logger.info(f"Memory ID: {config['memory_id']}, User ID: {config['user_id']}")
        
    except Exception as e:
        logger.error(f"Failed to initialize GraphMem: {e}")
        raise


async def shutdown_graphmem():
    """Cleanup GraphMem resources."""
    global _graphmem_instance, _serving_engine
    
    if _graphmem_instance:
        try:
            # Sync if using Turso cloud
            if hasattr(_graphmem_instance, '_graph_store'):
                store = _graphmem_instance._graph_store
                if hasattr(store, 'sync'):
                    store.sync()
        except Exception as e:
            logger.warning(f"Error during shutdown sync: {e}")
    
    _graphmem_instance = None
    _serving_engine = None
    logger.info("GraphMem shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    await initialize_graphmem()
    yield
    # Shutdown
    await shutdown_graphmem()


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration dictionary. If not provided,
                uses environment variables or defaults.
    
    Returns:
        Configured FastAPI application
    """
    
    # Update config if provided
    if config:
        DEFAULT_CONFIG.update(config)
    
    app = FastAPI(
        title="GraphMem OpenAI-Compatible API",
        description="""
        High-performance memory-augmented LLM API server with OpenAI compatibility.
        
        This server wraps GraphMem to provide:
        - Full OpenAI API compatibility
        - Memory-augmented responses using graph-based memory
        - Turso-powered fast retrieval
        - Streaming support
        
        Use any OpenAI-compatible client to interact with this API.
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ========================================================================
    # Health & Status Endpoints
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": DEFAULT_CONFIG["model_name"],
            "memory_id": DEFAULT_CONFIG["memory_id"],
            "timestamp": int(time.time())
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "GraphMem OpenAI-Compatible API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "chat_completions": "/v1/chat/completions",
                "completions": "/v1/completions",
                "models": "/v1/models",
                "embeddings": "/v1/embeddings",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    # ========================================================================
    # OpenAI API Endpoints
    # ========================================================================
    
    @app.get("/v1/models")
    async def list_models():
        """List available models - OpenAI compatible."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=DEFAULT_CONFIG["model_name"],
                    created=int(time.time()),
                    owned_by="graphmem"
                ),
                # Also expose the underlying LLM
                ModelInfo(
                    id=DEFAULT_CONFIG["llm_model"],
                    created=int(time.time()),
                    owned_by=DEFAULT_CONFIG["llm_provider"]
                )
            ]
        )
    
    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str):
        """Retrieve a specific model - OpenAI compatible."""
        if model_id in [DEFAULT_CONFIG["model_name"], DEFAULT_CONFIG["llm_model"]]:
            return ModelInfo(
                id=model_id,
                created=int(time.time()),
                owned_by="graphmem" if model_id == DEFAULT_CONFIG["model_name"] else DEFAULT_CONFIG["llm_provider"]
            )
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest,
        engine: GraphMemServingEngine = Depends(get_serving_engine)
    ):
        """
        Create chat completion - OpenAI compatible.
        
        Supports:
        - Standard chat completions
        - Streaming responses
        - Memory-augmented responses (unique to GraphMem)
        """
        try:
            result = await engine.create_chat_completion(request)
            
            if request.stream:
                # Return streaming response
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    }
                )
            else:
                return result
                
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=str(e),
                        type="server_error",
                        code="internal_error"
                    )
                ).model_dump()
            )
    
    @app.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest,
        engine: GraphMemServingEngine = Depends(get_serving_engine)
    ):
        """
        Create text completion (legacy endpoint) - OpenAI compatible.
        """
        try:
            result = await engine.create_completion(request)
            
            if request.stream:
                return StreamingResponse(
                    result,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                return result
                
        except Exception as e:
            logger.error(f"Completion error: {e}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=str(e),
                        type="server_error",
                        code="internal_error"
                    )
                ).model_dump()
            )
    
    @app.post("/v1/embeddings")
    async def create_embeddings(
        request: EmbeddingRequest,
        engine: GraphMemServingEngine = Depends(get_serving_engine)
    ):
        """
        Create embeddings - OpenAI compatible.
        
        Uses GraphMem's embedding provider to generate embeddings.
        """
        try:
            # Get embedding provider from graphmem (stored as _embeddings)
            embedding_provider = getattr(engine.graphmem, '_embeddings', None)
            
            if embedding_provider is None:
                raise HTTPException(status_code=503, detail="Embedding provider not configured")
            
            # Handle single or multiple inputs
            inputs = request.input if isinstance(request.input, list) else [request.input]
            
            embeddings = []
            total_tokens = 0
            
            for i, text in enumerate(inputs):
                if isinstance(text, str):
                    # Generate embedding - try different method names
                    if hasattr(embedding_provider, 'get_embedding'):
                        embedding = await asyncio.to_thread(
                            embedding_provider.get_embedding,
                            text
                        )
                    elif hasattr(embedding_provider, 'embed'):
                        embedding = await asyncio.to_thread(
                            embedding_provider.embed,
                            text
                        )
                    elif hasattr(embedding_provider, '__call__'):
                        embedding = await asyncio.to_thread(
                            embedding_provider,
                            text
                        )
                    else:
                        raise HTTPException(status_code=503, detail="Embedding provider has no embed method")
                    
                    embeddings.append(
                        EmbeddingData(
                            embedding=embedding,
                            index=i
                        )
                    )
                    total_tokens += len(text.split())  # Rough token estimate
            
            return EmbeddingResponse(
                data=embeddings,
                model=request.model or DEFAULT_CONFIG["embedding_model"],
                usage=EmbeddingUsage(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens
                )
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        message=str(e),
                        type="server_error",
                        code="internal_error"
                    )
                ).model_dump()
            )
    
    # ========================================================================
    # GraphMem-Specific Endpoints
    # ========================================================================
    
    @app.get("/v1/memory/stats")
    async def get_memory_stats(
        engine: GraphMemServingEngine = Depends(get_serving_engine)
    ):
        """Get GraphMem memory statistics."""
        try:
            graph = engine.graphmem.get_graph()
            return {
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
                "clusters": len(graph.get("clusters", [])),
                "serving_stats": engine.get_stats()
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}
    
    @app.post("/v1/memory/query")
    async def query_memory(
        query: str,
        engine: GraphMemServingEngine = Depends(get_serving_engine)
    ):
        """
        Query GraphMem directly (bypassing LLM generation).
        
        Useful for debugging and understanding what memory context
        would be used for a given query.
        """
        try:
            response = engine.graphmem.query(query)
            return {
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": [
                    {
                        "content": s.content,
                        "relevance": s.relevance_score,
                        "type": s.source_type
                    }
                    for s in response.sources
                ] if response.sources else []
            }
        except Exception as e:
            logger.error(f"Memory query error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    config: Dict[str, Any] = None,
    reload: bool = False,
    workers: int = 1,
):
    """
    Run the GraphMem OpenAI-compatible server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        config: Optional configuration dictionary
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    import uvicorn
    
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphMem OpenAI-Compatible API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # Config overrides
    parser.add_argument("--llm-provider", help="LLM provider")
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--turso-db-path", help="Turso database path")
    parser.add_argument("--turso-url", help="Turso cloud URL")
    parser.add_argument("--memory-id", help="Memory ID")
    parser.add_argument("--user-id", help="User ID")
    parser.add_argument("--model-name", help="Model name to report")
    
    args = parser.parse_args()
    
    # Build config from args
    config = {}
    if args.llm_provider:
        config["llm_provider"] = args.llm_provider
    if args.llm_model:
        config["llm_model"] = args.llm_model
    if args.turso_db_path:
        config["turso_db_path"] = args.turso_db_path
    if args.turso_url:
        config["turso_url"] = args.turso_url
    if args.memory_id:
        config["memory_id"] = args.memory_id
    if args.user_id:
        config["user_id"] = args.user_id
    if args.model_name:
        config["model_name"] = args.model_name
    
    run_server(
        host=args.host,
        port=args.port,
        config=config if config else None,
        reload=args.reload,
        workers=args.workers
    )


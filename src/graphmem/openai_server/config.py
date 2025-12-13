"""
Configuration for GraphMem OpenAI Server

This module provides the configuration loading and validation for the server.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server configuration model."""
    
    # LLM Configuration
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_LLM_PROVIDER", "azure_openai"),
        description="LLM provider (azure_openai, openai, openrouter, ollama)"
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_KEY", ""),
        description="LLM API key"
    )
    llm_api_base: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_LLM_API_BASE", ""),
        description="LLM API base URL"
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_LLM_MODEL", "gpt-4.1-mini"),
        description="LLM model name"
    )
    
    # Embedding Configuration
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_PROVIDER", "azure_openai"),
        description="Embedding provider"
    )
    embedding_api_key: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_API_KEY", ""),
        description="Embedding API key"
    )
    embedding_api_base: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_API_BASE", ""),
        description="Embedding API base URL"
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_EMBEDDING_MODEL", "text-embedding-3-small"),
        description="Embedding model name"
    )
    
    # Turso Configuration
    turso_db_path: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_TURSO_DB_PATH", "graphmem_local.db"),
        description="Local Turso database path"
    )
    turso_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_TURSO_URL") or None,
        description="Turso cloud URL"
    )
    turso_auth_token: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_TURSO_AUTH_TOKEN") or None,
        description="Turso auth token"
    )
    
    # Memory Configuration
    memory_id: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_MEMORY_ID", "benchmark-memory"),
        description="Memory ID for persistence"
    )
    user_id: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_USER_ID", "benchmark-user"),
        description="User ID for persistence"
    )
    
    # Server Configuration
    model_name: str = Field(
        default_factory=lambda: os.getenv("GRAPHMEM_MODEL_NAME", "graphmem-turso"),
        description="Model name to report in API responses"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for server initialization."""
        return {
            "llm_provider": self.llm_provider,
            "llm_api_key": self.llm_api_key,
            "llm_api_base": self.llm_api_base,
            "llm_model": self.llm_model,
            "embedding_provider": self.embedding_provider,
            "embedding_api_key": self.embedding_api_key,
            "embedding_api_base": self.embedding_api_base,
            "embedding_model": self.embedding_model,
            "turso_db_path": self.turso_db_path,
            "turso_url": self.turso_url,
            "turso_auth_token": self.turso_auth_token,
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "model_name": self.model_name,
        }


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> ServerConfig:
    """
    Load server configuration.
    
    Args:
        config_path: Optional path to JSON config file
        overrides: Optional dictionary of overrides
        
    Returns:
        ServerConfig instance
    """
    import json
    
    config_dict = {}
    
    # Load from file if provided
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # Apply overrides
    if overrides:
        config_dict.update(overrides)
    
    return ServerConfig(**config_dict)


# Hardcoded configuration for benchmarking
# Modify these values for your specific use case
BENCHMARK_CONFIG = ServerConfig(
    llm_provider="azure_openai",
    llm_api_key="YOUR_API_KEY",
    llm_api_base="https://your-endpoint.openai.azure.com/",
    llm_model="gpt-4.1-mini",
    embedding_provider="azure_openai",
    embedding_api_key="YOUR_API_KEY",
    embedding_api_base="https://your-endpoint.openai.azure.com/",
    embedding_model="text-embedding-3-small",
    turso_db_path="graphmem_local.db",
    turso_url="libsql://your-db.turso.io",
    turso_auth_token="YOUR_TURSO_TOKEN",
    memory_id="benchmark-memory",
    user_id="benchmark-user",
    model_name="graphmem-turso",
)


"""
GraphMem Embedding Providers

Abstraction layer for embedding generation supporting multiple providers.

Supports:
- OpenAI
- Azure OpenAI
- OpenRouter (via OpenAI-compatible API)
- Together AI (via OpenAI-compatible API)
- Any OpenAI-compatible embedding API
- Local models (sentence-transformers)
"""

from __future__ import annotations
import logging
import os
from typing import List, Optional
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    Multi-provider embedding abstraction supporting any OpenAI-compatible API.
    
    Supported Providers:
    - azure_openai: Azure OpenAI Service
    - openai: OpenAI API
    - openai_compatible: Any OpenAI-compatible API (OpenRouter, Together, etc.)
    - local: Local sentence-transformers models
    
    Examples:
        # OpenAI
        embeddings = EmbeddingProvider(
            provider="openai",
            api_key="sk-...",
            model="text-embedding-3-small"
        )
        
        # Azure OpenAI
        embeddings = EmbeddingProvider(
            provider="azure_openai",
            api_key="...",
            api_base="https://your-resource.openai.azure.com/",
            deployment="text-embedding-3-small"
        )
        
        # OpenRouter
        embeddings = EmbeddingProvider(
            provider="openai_compatible",
            api_key="sk-or-v1-...",
            api_base="https://openrouter.ai/api/v1",
            model="openai/text-embedding-3-small"
        )
        
        # Together AI
        embeddings = EmbeddingProvider(
            provider="openai_compatible",
            api_key="...",
            api_base="https://api.together.xyz/v1",
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        
        # Local (sentence-transformers)
        embeddings = EmbeddingProvider(
            provider="local",
            model="all-MiniLM-L6-v2"
        )
    """
    
    def __init__(
        self,
        provider: str = "azure_openai",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
        deployment: Optional[str] = None,
        cache=None,
        **kwargs,
    ):
        """
        Initialize embedding provider.
        
        Args:
            provider: Provider name (azure_openai, openai, openai_compatible, local)
            api_key: API key
            api_base: API base URL (required for openai_compatible)
            api_version: API version (Azure only)
            model: Model name
            deployment: Deployment name (Azure only)
            cache: Optional cache for embeddings
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.model = model
        self.deployment = deployment
        self.cache = cache
        self.kwargs = kwargs
        
        self._client = None
        self._local_model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client."""
        if self.provider == "azure_openai":
            self._init_azure_openai()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "openai_compatible":
            self._init_openai_compatible()
        elif self.provider == "local":
            self._init_local()
        else:
            raise ValueError(
                f"Unsupported embedding provider: {self.provider}. "
                f"Supported: azure_openai, openai, openai_compatible, local"
            )
    
    def _init_azure_openai(self):
        """Initialize Azure OpenAI client."""
        try:
            from openai import AzureOpenAI
            
            self._client = AzureOpenAI(
                api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=self.api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            self.model = self.deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.api_base,  # None uses default
            )
            self.model = self.model or "text-embedding-3-small"
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_openai_compatible(self):
        """
        Initialize OpenAI-compatible client for third-party providers.
        
        Works with:
        - OpenRouter: https://openrouter.ai/api/v1
        - Together AI: https://api.together.xyz/v1
        - Anyscale: https://api.endpoints.anyscale.com/v1
        - And any other OpenAI-compatible embedding endpoint
        """
        try:
            from openai import OpenAI
            
            if not self.api_base:
                raise ValueError(
                    "api_base is required for openai_compatible provider. "
                    "Example: api_base='https://openrouter.ai/api/v1'"
                )
            
            if not self.api_key:
                raise ValueError("api_key is required for openai_compatible provider")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            
            if not self.model:
                raise ValueError(
                    "model is required for openai_compatible provider. "
                    "Example: model='openai/text-embedding-3-small'"
                )
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_local(self):
        """Initialize local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = self.model or "all-MiniLM-L6-v2"
            self._local_model = SentenceTransformer(self.model)
            
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return []
        
        text = text.replace("\n", " ").strip()
        
        # Check cache
        if self.cache:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cached = self.cache.get_embedding(text_hash)
            if cached:
                return cached
        
        embedding = self._generate_embedding(text)
        
        # Cache result
        if self.cache and embedding:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            self.cache.cache_embedding(text_hash, embedding)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        # Clean texts
        texts = [t.replace("\n", " ").strip() for t in texts if t and t.strip()]
        
        if not texts:
            return []
        
        if self.provider == "local":
            return self._local_embed_batch(texts)
        else:
            return self._api_embed_batch(texts)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding."""
        if self.provider == "local":
            embedding = self._local_model.encode(text)
            return embedding.tolist()
        else:
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=[text],
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                return []
    
    def _local_embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        try:
            embeddings = self._local_model.encode(texts)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Local embedding batch error: {e}")
            return []
    
    def _api_embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings using API with batching."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"API embedding batch error: {e}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])
        
        return all_embeddings
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0
        
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding_provider(
    provider: str = "openai",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    model: Optional[str] = None,
    deployment: Optional[str] = None,
    cache=None,
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider.
    
    Args:
        provider: Provider name:
            - "openai": OpenAI API
            - "azure_openai": Azure OpenAI Service
            - "openai_compatible": Any OpenAI-compatible API
            - "local": Local sentence-transformers
        api_key: API key
        api_base: API base URL (required for openai_compatible)
        api_version: API version (Azure only)
        model: Model name
        deployment: Deployment name (Azure only)
        cache: Optional cache for embeddings
    
    Returns:
        Configured EmbeddingProvider instance
    
    Examples:
        # OpenAI
        emb = get_embedding_provider("openai", api_key="sk-...")
        
        # OpenRouter
        emb = get_embedding_provider(
            "openai_compatible",
            api_key="sk-or-v1-...",
            api_base="https://openrouter.ai/api/v1",
            model="openai/text-embedding-3-small"
        )
        
        # Local
        emb = get_embedding_provider("local", model="all-MiniLM-L6-v2")
    """
    return EmbeddingProvider(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        model=model,
        deployment=deployment,
        cache=cache,
        **kwargs,
    )


# Convenience functions for common providers
def openrouter_embeddings(api_key: str, model: str = "openai/text-embedding-3-small") -> EmbeddingProvider:
    """Create an OpenRouter embedding provider."""
    return EmbeddingProvider(
        provider="openai_compatible",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        model=model,
    )


def together_embeddings(api_key: str, model: str = "togethercomputer/m2-bert-80M-8k-retrieval") -> EmbeddingProvider:
    """Create a Together AI embedding provider."""
    return EmbeddingProvider(
        provider="openai_compatible",
        api_key=api_key,
        api_base="https://api.together.xyz/v1",
        model=model,
    )

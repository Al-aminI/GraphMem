"""
GraphMem LLM Providers

Abstraction layer for LLM interactions supporting multiple providers.

Supports:
- OpenAI
- Azure OpenAI  
- Anthropic Claude
- OpenRouter (100+ models)
- Together AI
- Groq
- Any OpenAI-compatible API
- Local models (Ollama)
"""

from __future__ import annotations
import logging
import os
from typing import List, Dict, Any, Optional, Literal, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Generate completion for a prompt."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Generate chat completion."""
        pass


class LLMProvider(BaseLLM):
    """
    Multi-provider LLM abstraction supporting any OpenAI-compatible API.
    
    Supported Providers:
    - azure_openai: Azure OpenAI Service
    - openai: OpenAI API
    - openai_compatible: Any OpenAI-compatible API (OpenRouter, Together, Groq, etc.)
    - anthropic: Anthropic Claude
    - ollama: Local Ollama models
    
    Examples:
        # OpenAI
        llm = LLMProvider(provider="openai", api_key="sk-...", model="gpt-4o")
        
        # Azure OpenAI
        llm = LLMProvider(
            provider="azure_openai",
            api_key="...",
            api_base="https://your-resource.openai.azure.com/",
            deployment="gpt-4"
        )
        
        # OpenRouter (100+ models)
        llm = LLMProvider(
            provider="openai_compatible",
            api_key="sk-or-v1-...",
            api_base="https://openrouter.ai/api/v1",
            model="google/gemini-2.5-flash"
        )
        
        # Together AI
        llm = LLMProvider(
            provider="openai_compatible",
            api_key="...",
            api_base="https://api.together.xyz/v1",
            model="meta-llama/Llama-3-70b-chat-hf"
        )
        
        # Groq
        llm = LLMProvider(
            provider="openai_compatible",
            api_key="gsk_...",
            api_base="https://api.groq.com/openai/v1",
            model="llama-3.1-70b-versatile"
        )
        
        # Local Ollama
        llm = LLMProvider(provider="ollama", model="llama3.2")
    """
    
    def __init__(
        self,
        provider: str = "azure_openai",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None,
        deployment: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize LLM provider.
        
        Args:
            provider: Provider name (azure_openai, openai, openai_compatible, anthropic, ollama)
            api_key: API key (or from env)
            api_base: API base URL (required for openai_compatible)
            api_version: API version (Azure only)
            model: Model name/identifier
            deployment: Deployment name (Azure only)
            **kwargs: Additional provider-specific options
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.model = model
        self.deployment = deployment
        self.kwargs = kwargs
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client."""
        if self.provider == "azure_openai":
            self._init_azure_openai()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "openai_compatible":
            self._init_openai_compatible()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. "
                           f"Supported: azure_openai, openai, openai_compatible, anthropic, ollama")
    
    def _init_azure_openai(self):
        """Initialize Azure OpenAI client."""
        try:
            from openai import AzureOpenAI
            
            self._client = AzureOpenAI(
                api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=self.api_base or os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            self.model = self.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.api_base,  # None uses default OpenAI endpoint
            )
            self.model = self.model or "gpt-4o"
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_openai_compatible(self):
        """
        Initialize OpenAI-compatible client for third-party providers.
        
        Works with:
        - OpenRouter: https://openrouter.ai/api/v1
        - Together AI: https://api.together.xyz/v1  
        - Groq: https://api.groq.com/openai/v1
        - Anyscale: https://api.endpoints.anyscale.com/v1
        - Fireworks: https://api.fireworks.ai/inference/v1
        - Perplexity: https://api.perplexity.ai
        - DeepInfra: https://api.deepinfra.com/v1/openai
        - Mistral: https://api.mistral.ai/v1
        - And any other OpenAI-compatible endpoint
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
                    "Example: model='google/gemini-2.5-flash'"
                )
            
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            self._client = anthropic.Anthropic(
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
            self.model = self.model or "claude-3-5-sonnet-20241022"
            
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        self.api_base = self.api_base or "http://localhost:11434"
        self.model = self.model or "llama3.2"
    
    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Generate completion."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Generate chat completion."""
        if self.provider in ("azure_openai", "openai", "openai_compatible"):
            return self._openai_chat(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._anthropic_chat(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._ollama_chat(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _openai_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """OpenAI/Azure chat completion."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    def _anthropic_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Anthropic chat completion."""
        try:
            # Extract system message
            system = None
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system = msg["content"]
                else:
                    chat_messages.append(msg)
            
            kwargs = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": max_tokens,
            }
            if system:
                kwargs["system"] = system
            
            response = self._client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Ollama chat completion."""
        import requests
        
        try:
            response = requests.post(
                f"{self.api_base}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise
    
    def analyze_image(
        self,
        image_b64: str,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        """Analyze an image using vision model."""
        if self.provider in ("azure_openai", "openai", "openai_compatible"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ]
            return self._openai_chat(messages, 0.1, 4000)
        elif self.provider == "anthropic":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            return self._anthropic_chat(messages, 0.1, 4000)
        else:
            raise ValueError(f"Vision not supported for provider: {self.provider}")


def get_llm_provider(
    provider: str = "openai",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    model: Optional[str] = None,
    deployment: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.
    
    Args:
        provider: Provider name:
            - "openai": OpenAI API
            - "azure_openai": Azure OpenAI Service
            - "openai_compatible": Any OpenAI-compatible API
            - "anthropic": Anthropic Claude
            - "ollama": Local Ollama models
        api_key: API key
        api_base: API base URL (required for openai_compatible)
        api_version: API version (Azure only)
        model: Model name
        deployment: Deployment name (Azure only)
    
    Returns:
        Configured LLMProvider instance
    
    Examples:
        # OpenAI
        llm = get_llm_provider("openai", api_key="sk-...")
        
        # OpenRouter (access 100+ models)
        llm = get_llm_provider(
            "openai_compatible",
            api_key="sk-or-v1-...",
            api_base="https://openrouter.ai/api/v1",
            model="google/gemini-2.5-flash"
        )
        
        # Groq (fast inference)
        llm = get_llm_provider(
            "openai_compatible",
            api_key="gsk_...",
            api_base="https://api.groq.com/openai/v1",
            model="llama-3.1-70b-versatile"
        )
        
        # Together AI
        llm = get_llm_provider(
            "openai_compatible",
            api_key="...",
            api_base="https://api.together.xyz/v1",
            model="meta-llama/Llama-3-70b-chat-hf"
        )
    """
    return LLMProvider(
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        model=model,
        deployment=deployment,
        **kwargs,
    )


# Convenience aliases for common providers
def openrouter(api_key: str, model: str, **kwargs) -> LLMProvider:
    """Create an OpenRouter provider."""
    return LLMProvider(
        provider="openai_compatible",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        model=model,
        **kwargs,
    )


def groq(api_key: str, model: str = "llama-3.1-70b-versatile", **kwargs) -> LLMProvider:
    """Create a Groq provider."""
    return LLMProvider(
        provider="openai_compatible",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1",
        model=model,
        **kwargs,
    )


def together(api_key: str, model: str, **kwargs) -> LLMProvider:
    """Create a Together AI provider."""
    return LLMProvider(
        provider="openai_compatible",
        api_key=api_key,
        api_base="https://api.together.xyz/v1",
        model=model,
        **kwargs,
    )

"""
GraphMem LLM Providers

Abstraction layer for LLM interactions supporting multiple providers.
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
    Multi-provider LLM abstraction.
    
    Supports:
    - Azure OpenAI
    - OpenAI
    - Anthropic Claude
    - Local models (Ollama)
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
            provider: Provider name (azure_openai, openai, anthropic, ollama)
            api_key: API key (or from env)
            api_base: API base URL
            api_version: API version (Azure)
            model: Model name
            deployment: Deployment name (Azure)
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
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
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
                base_url=self.api_base,
            )
            self.model = self.model or "gpt-4o"
            
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
        if self.provider in ("azure_openai", "openai"):
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
        if self.provider in ("azure_openai", "openai"):
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
        provider: Provider name (azure_openai, openai, anthropic, ollama)
        api_key: API key
        api_base: API base URL
        api_version: API version (Azure)
        model: Model name
        deployment: Deployment name (Azure)
    
    Returns:
        Configured LLMProvider instance
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

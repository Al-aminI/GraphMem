# GraphMem OpenAI-Compatible API Server
"""
OpenAI-compatible API server for GraphMem.

This module provides a FastAPI server that exposes GraphMem's memory-augmented
responses through an OpenAI-compatible API, allowing seamless integration with
existing AI applications.
"""

from .protocol import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelsResponse,
)
from .server import create_app, run_server

__all__ = [
    "ChatMessage",
    "ChatCompletionRequest", 
    "ChatCompletionResponse",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionStreamResponse",
    "ChatCompletionStreamChoice",
    "CompletionRequest",
    "CompletionResponse",
    "ModelInfo",
    "ModelsResponse",
    "create_app",
    "run_server",
]


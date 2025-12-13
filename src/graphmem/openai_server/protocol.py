"""
OpenAI API Protocol Models

Pydantic models that exactly match the OpenAI API specification for
seamless compatibility with existing AI applications.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
import uuid
import time


# ============================================================================
# Chat Completion Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message model following OpenAI specification"""
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    """Chat completion request model - exactly matches OpenAI spec"""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    
    # GraphMem-specific options (optional)
    memory_id: Optional[str] = None
    user_id: Optional[str] = None
    use_memory: Optional[bool] = True

    class Config:
        extra = "allow"


class ChatCompletionUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Optional detailed breakdown (OpenAI extension)
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model"""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model - exactly matches OpenAI spec"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None
    
    # GraphMem-specific (optional)
    memory_context: Optional[Dict[str, Any]] = None


# ============================================================================
# Streaming Models
# ============================================================================

class ChatCompletionStreamDelta(BaseModel):
    """Delta object for streaming responses"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionStreamChoice(BaseModel):
    """Chat completion stream choice model"""
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response model"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[ChatCompletionUsage] = None


# ============================================================================
# Legacy Completion Models (for /v1/completions)
# ============================================================================

class CompletionRequest(BaseModel):
    """Legacy completion request model"""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    suffix: Optional[str] = None
    seed: Optional[int] = None

    class Config:
        extra = "allow"


class CompletionChoice(BaseModel):
    """Legacy completion choice"""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    """Legacy completion response"""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


# ============================================================================
# Model Information
# ============================================================================

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "graphmem"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for /v1/models endpoint"""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail model"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model - matches OpenAI error format"""
    error: ErrorDetail


# ============================================================================
# Embeddings Models (for completeness)
# ============================================================================

class EmbeddingRequest(BaseModel):
    """Embedding request model"""
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """Single embedding data"""
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Embedding usage info"""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Embedding response model"""
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


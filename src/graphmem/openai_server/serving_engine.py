"""
GraphMem Serving Engine

Core serving logic that wraps GraphMem with OpenAI-compatible interfaces.
Handles memory-augmented generation with configurable LLM backends.
"""

import json
import time
import uuid
import logging
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List

from .protocol import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ErrorResponse,
    ErrorDetail,
)

logger = logging.getLogger(__name__)


class GraphMemServingEngine:
    """
    Serving engine that wraps GraphMem for OpenAI-compatible responses.
    
    This engine:
    1. Receives OpenAI-format requests
    2. Queries GraphMem for relevant memory context
    3. Augments the prompt with memory
    4. Generates responses using the configured LLM
    5. Returns OpenAI-format responses
    """
    
    def __init__(
        self,
        graphmem_instance,
        model_name: str = "graphmem-turso",
        system_fingerprint: str = None,
    ):
        """
        Initialize the serving engine.
        
        Args:
            graphmem_instance: Initialized GraphMem instance with Turso backend
            model_name: Name to report as the model
            system_fingerprint: Optional system fingerprint for responses
        """
        self.graphmem = graphmem_instance
        self.model_name = model_name
        self.system_fingerprint = system_fingerprint or f"fp_{uuid.uuid4().hex[:12]}"
        self._request_count = 0
        
        logger.info(f"GraphMemServingEngine initialized with model: {model_name}")
    
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse | AsyncGenerator[str, None]:
        """
        Create chat completion following OpenAI API specification.
        
        Args:
            request: OpenAI-format chat completion request
            
        Returns:
            ChatCompletionResponse or async generator for streaming
        """
        try:
            self._request_count += 1
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            
            # Extract the user's query from messages
            user_query = self._extract_user_query(request.messages)
            system_prompt = self._extract_system_prompt(request.messages)
            
            # Query GraphMem for relevant context
            memory_context = None
            if request.use_memory and user_query:
                try:
                    memory_response = self.graphmem.query(user_query)
                    memory_context = {
                        "answer": memory_response.answer,
                        "confidence": memory_response.confidence,
                        "sources": [
                            {
                                "content": s.content,
                                "relevance": s.relevance_score,
                                "type": s.source_type
                            }
                            for s in memory_response.sources[:5]  # Top 5 sources
                        ] if memory_response.sources else []
                    }
                except Exception as e:
                    logger.warning(f"Memory query failed: {e}")
                    memory_context = None
            
            # Build the augmented prompt
            augmented_messages = self._augment_with_memory(
                request.messages,
                memory_context,
                system_prompt
            )
            
            # Generate response
            if request.stream:
                return self._create_streaming_response(
                    request_id=request_id,
                    messages=augmented_messages,
                    memory_context=memory_context,
                    request=request,
                    original_messages=request.messages
                )
            else:
                return await self._create_completion_response(
                    request_id=request_id,
                    messages=augmented_messages,
                    memory_context=memory_context,
                    request=request,
                    original_messages=request.messages
                )
                
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise
    
    def _extract_user_query(self, messages: List[ChatMessage]) -> str:
        """Extract the last user message as the query."""
        for message in reversed(messages):
            if message.role == "user" and message.content:
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, list):
                    # Handle multi-modal content
                    for part in message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""
    
    def _extract_system_prompt(self, messages: List[ChatMessage]) -> str:
        """Extract system prompt if present."""
        for message in messages:
            if message.role == "system" and message.content:
                return message.content if isinstance(message.content, str) else ""
        return ""
    
    def _augment_with_memory(
        self,
        messages: List[ChatMessage],
        memory_context: Optional[Dict[str, Any]],
        system_prompt: str
    ) -> List[ChatMessage]:
        """Augment messages with memory context."""
        augmented = []
        
        # Build enhanced system prompt with memory
        memory_section = ""
        if memory_context and memory_context.get("sources"):
            memory_section = "\n\n## Relevant Memory Context:\n"
            for i, source in enumerate(memory_context["sources"], 1):
                memory_section += f"{i}. {source['content']} (relevance: {source['relevance']:.2f})\n"
            
            if memory_context.get("answer"):
                memory_section += f"\n**Memory Summary:** {memory_context['answer']}\n"
        
        enhanced_system = system_prompt or "You are a helpful AI assistant with access to memory."
        if memory_section:
            enhanced_system += memory_section
            enhanced_system += "\nUse the above memory context to inform your response when relevant."
        
        # Add enhanced system message
        augmented.append(ChatMessage(role="system", content=enhanced_system))
        
        # Add remaining messages (skip original system)
        for msg in messages:
            if msg.role != "system":
                augmented.append(msg)
        
        return augmented
    
    async def _create_completion_response(
        self,
        request_id: str,
        messages: List[ChatMessage],
        memory_context: Optional[Dict[str, Any]],
        request: ChatCompletionRequest,
        original_messages: List[ChatMessage] = None
    ) -> ChatCompletionResponse:
        """Create non-streaming completion response."""
        
        # Use GraphMem's LLM to generate response
        # Build prompt from messages
        prompt = self._messages_to_prompt(messages)
        
        # Generate using GraphMem's LLM provider
        try:
            # Use the LLM directly from graphmem, passing messages for chat method
            response_text = await self._generate_with_llm(prompt, request, messages)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response_text = f"I apologize, but I encountered an error generating a response: {str(e)}"
        
        # Estimate token counts (rough approximation)
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        completion_tokens = len(response_text.split()) * 1.3
        
        return ChatCompletionResponse(
            id=request_id,
            model=request.model or self.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens)
            ),
            system_fingerprint=self.system_fingerprint,
            memory_context=memory_context
        )
    
    async def _create_streaming_response(
        self,
        request_id: str,
        messages: List[ChatMessage],
        memory_context: Optional[Dict[str, Any]],
        request: ChatCompletionRequest,
        original_messages: List[ChatMessage] = None
    ) -> AsyncGenerator[str, None]:
        """Create streaming completion response as an async generator."""
        created = int(time.time())
        model = request.model or self.model_name
        
        # Send initial chunk with role
        initial_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant"),
                    finish_reason=None
                )
            ],
            system_fingerprint=self.system_fingerprint
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Generate response
        prompt = self._messages_to_prompt(messages)
        
        try:
            response_text = await self._generate_with_llm(prompt, request, messages)
            
            # Stream the response word by word for demo
            # In production, you'd use actual streaming from the LLM
            words = response_text.split()
            for i, word in enumerate(words):
                content = word + (" " if i < len(words) - 1 else "")
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta=ChatCompletionStreamDelta(content=content),
                            finish_reason=None
                        )
                    ],
                    system_fingerprint=self.system_fingerprint
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            error_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(content=f"Error: {str(e)}"),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
        
        # Send final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(),
                    finish_reason="stop"
                )
            ],
            system_fingerprint=self.system_fingerprint
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a prompt string."""
        # For chat-based LLMs, we use the messages directly
        # This method is mainly for fallback/legacy completions
        prompt_parts = []
        
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    def _messages_to_chat_format(self, messages: List[ChatMessage]) -> List[dict]:
        """Convert ChatMessage objects to OpenAI chat format."""
        formatted = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else ""
            formatted.append({
                "role": msg.role,
                "content": content
            })
        return formatted
    
    async def _generate_with_llm(
        self,
        prompt: str,
        request: ChatCompletionRequest,
        messages: List[ChatMessage] = None
    ) -> str:
        """Generate response using GraphMem's LLM provider."""
        
        # Access GraphMem's LLM provider (stored as _llm)
        llm_provider = getattr(self.graphmem, '_llm', None)
        
        if llm_provider is None:
            # Fallback to memory-only response
            try:
                response = self.graphmem.query(prompt)
                return response.answer if response.answer else "No relevant information found in memory."
            except Exception as e:
                return f"LLM provider not configured and memory query failed: {str(e)}"
        
        try:
            max_tokens = request.max_tokens or request.max_completion_tokens or 1024
            temperature = request.temperature or 0.7
            
            # Prefer chat method if available and messages provided
            if messages and hasattr(llm_provider, 'chat'):
                chat_messages = self._messages_to_chat_format(messages)
                response = await asyncio.to_thread(
                    llm_provider.chat,
                    chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            
            # Try different methods based on provider capabilities
            if hasattr(llm_provider, 'generate'):
                response = await asyncio.to_thread(
                    llm_provider.generate,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            elif hasattr(llm_provider, 'complete'):
                response = await asyncio.to_thread(
                    llm_provider.complete,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            elif hasattr(llm_provider, '__call__'):
                response = await asyncio.to_thread(
                    llm_provider,
                    prompt
                )
                return str(response)
            else:
                # Fallback: try to use the query method of graphmem
                response = self.graphmem.query(prompt)
                return response.answer if response.answer else "No relevant information found."
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Fallback to memory-only response
            try:
                response = self.graphmem.query(prompt)
                return response.answer if response.answer else f"Memory query returned no results. LLM error: {str(e)}"
            except Exception as query_error:
                return f"Both LLM and memory query failed. LLM error: {str(e)}, Memory error: {str(query_error)}"
    
    async def create_completion(
        self,
        request: CompletionRequest
    ) -> CompletionResponse | AsyncGenerator[str, None]:
        """Create legacy completion (for /v1/completions endpoint)."""
        
        # Convert to chat format and use chat completion
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        if request.stream:
            # Return streaming generator adapted for completion format
            chat_stream = await self.create_chat_completion(chat_request)
            return self._adapt_to_completion_stream(chat_stream, request)
        else:
            chat_response = await self.create_chat_completion(chat_request)
            
            return CompletionResponse(
                id=chat_response.id.replace("chatcmpl-", "cmpl-"),
                model=chat_response.model,
                choices=[
                    CompletionChoice(
                        text=chat_response.choices[0].message.content,
                        index=0,
                        finish_reason=chat_response.choices[0].finish_reason
                    )
                ],
                usage=chat_response.usage,
                system_fingerprint=chat_response.system_fingerprint
            )
    
    async def _adapt_to_completion_stream(
        self,
        chat_stream: AsyncGenerator,
        request: CompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Adapt chat streaming to completion streaming format."""
        async for chunk in chat_stream:
            # Parse and convert chunk
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    # Convert to completion format
                    if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                        completion_chunk = {
                            "id": data["id"].replace("chatcmpl-", "cmpl-"),
                            "object": "text_completion",
                            "created": data["created"],
                            "model": data["model"],
                            "choices": [{
                                "text": data["choices"][0]["delta"]["content"],
                                "index": 0,
                                "finish_reason": data["choices"][0].get("finish_reason")
                            }]
                        }
                        yield f"data: {json.dumps(completion_chunk)}\n\n"
                except:
                    yield chunk
            else:
                yield chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serving engine statistics."""
        return {
            "model_name": self.model_name,
            "request_count": self._request_count,
            "system_fingerprint": self.system_fingerprint,
            "graphmem_stats": self.graphmem.get_graph() if hasattr(self.graphmem, 'get_graph') else None
        }


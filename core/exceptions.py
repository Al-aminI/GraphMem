"""
GraphMem Exception Types

Production-grade exception hierarchy for comprehensive error handling.
All exceptions include context, recovery suggestions, and debugging information.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class GraphMemError(Exception):
    """
    Base exception for all GraphMem errors.
    
    Provides rich error context for debugging and monitoring in production.
    
    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        context: Additional error context
        timestamp: When the error occurred
        recoverable: Whether automatic recovery is possible
        suggestions: List of recovery suggestions
    """
    
    def __init__(
        self,
        message: str,
        code: str = "GRAPHMEM_ERROR",
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        suggestions: Optional[list] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        self.recoverable = recoverable
        self.suggestions = suggestions or []
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        base = f"[{self.code}] {self.message}"
        if self.context:
            base += f" | Context: {self.context}"
        if self.suggestions:
            base += f" | Suggestions: {', '.join(self.suggestions)}"
        return base
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.code!r}, {self.message!r})"


class ConfigurationError(GraphMemError):
    """
    Raised when GraphMem configuration is invalid or incomplete.
    
    Common causes:
    - Missing required environment variables
    - Invalid connection strings
    - Incompatible configuration options
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[Any] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        if expected:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = str(actual)
        
        suggestions = kwargs.pop("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check environment variables",
                "Verify configuration file",
                "Review GraphMem documentation",
            ]
        
        super().__init__(
            message=message,
            code="CONFIG_ERROR",
            context=context,
            recoverable=False,
            suggestions=suggestions,
            **kwargs,
        )


class StorageError(GraphMemError):
    """
    Raised when storage operations fail.
    
    Covers Neo4j, Redis, and any other storage backends.
    Includes retry logic hints and connection recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        storage_type: str = "unknown",
        operation: Optional[str] = None,
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context["storage_type"] = storage_type
        if operation:
            context["operation"] = operation
        if retry_after:
            context["retry_after_seconds"] = retry_after
        
        suggestions = kwargs.pop("suggestions", [])
        if not suggestions:
            suggestions = [
                f"Check {storage_type} connection",
                "Verify credentials and network access",
                "Review storage capacity and limits",
            ]
        
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            context=context,
            recoverable=retry_after is not None,
            suggestions=suggestions,
            **kwargs,
        )


class IngestionError(GraphMemError):
    """
    Raised when document/data ingestion fails.
    
    Covers document parsing, entity extraction, and graph construction.
    """
    
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        stage: Optional[str] = None,
        partial_success: bool = False,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if document_id:
            context["document_id"] = document_id
        if stage:
            context["stage"] = stage
        context["partial_success"] = partial_success
        
        suggestions = kwargs.pop("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check document format and encoding",
                "Verify LLM API availability",
                "Review document size limits",
            ]
        
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            context=context,
            recoverable=partial_success,
            suggestions=suggestions,
            **kwargs,
        )


class QueryError(GraphMemError):
    """
    Raised when memory queries fail.
    
    Covers semantic search, graph traversal, and context retrieval.
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        memory_id: Optional[str] = None,
        no_results: bool = False,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if query:
            context["query"] = query[:200] + "..." if len(query) > 200 else query
        if memory_id:
            context["memory_id"] = memory_id
        context["no_results"] = no_results
        
        suggestions = kwargs.pop("suggestions", [])
        if not suggestions:
            if no_results:
                suggestions = [
                    "Try a broader query",
                    "Ingest more relevant documents",
                    "Check if memory contains related topics",
                ]
            else:
                suggestions = [
                    "Check query syntax",
                    "Verify memory state is healthy",
                    "Review embedding model availability",
                ]
        
        super().__init__(
            message=message,
            code="QUERY_ERROR",
            context=context,
            recoverable=no_results,
            suggestions=suggestions,
            **kwargs,
        )


class EvolutionError(GraphMemError):
    """
    Raised when memory evolution operations fail.
    
    Covers consolidation, decay, rehydration, and self-improvement.
    """
    
    def __init__(
        self,
        message: str,
        evolution_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        rollback_available: bool = False,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if evolution_type:
            context["evolution_type"] = evolution_type
        if memory_id:
            context["memory_id"] = memory_id
        context["rollback_available"] = rollback_available
        
        suggestions = kwargs.pop("suggestions", [])
        if not suggestions:
            suggestions = [
                "Check memory consistency",
                "Review evolution parameters",
                "Consider rollback if available",
            ]
        
        super().__init__(
            message=message,
            code="EVOLUTION_ERROR",
            context=context,
            recoverable=rollback_available,
            suggestions=suggestions,
            **kwargs,
        )


class ExtractionError(GraphMemError):
    """
    Raised when entity/relationship extraction fails.
    """
    
    def __init__(
        self,
        message: str,
        extractor_type: Optional[str] = None,
        chunk_id: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if extractor_type:
            context["extractor_type"] = extractor_type
        if chunk_id:
            context["chunk_id"] = chunk_id
        
        super().__init__(
            message=message,
            code="EXTRACTION_ERROR",
            context=context,
            recoverable=True,
            suggestions=["Check LLM response format", "Review extraction prompts"],
            **kwargs,
        )


class EmbeddingError(GraphMemError):
    """
    Raised when embedding operations fail.
    """
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if model:
            context["model"] = model
        if text_length:
            context["text_length"] = text_length
        
        super().__init__(
            message=message,
            code="EMBEDDING_ERROR",
            context=context,
            recoverable=True,
            suggestions=["Check embedding API availability", "Verify text length limits"],
            **kwargs,
        )


class CommunityError(GraphMemError):
    """
    Raised when community detection/management fails.
    """
    
    def __init__(
        self,
        message: str,
        community_id: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        if community_id is not None:
            context["community_id"] = community_id
        
        super().__init__(
            message=message,
            code="COMMUNITY_ERROR",
            context=context,
            recoverable=True,
            suggestions=["Rebuild communities", "Check graph connectivity"],
            **kwargs,
        )


class RateLimitError(GraphMemError):
    """
    Raised when rate limits are exceeded.
    """
    
    def __init__(
        self,
        message: str,
        service: str = "unknown",
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context["service"] = service
        if retry_after:
            context["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            code="RATE_LIMIT_ERROR",
            context=context,
            recoverable=True,
            suggestions=[f"Wait {retry_after or 60} seconds before retrying"],
            **kwargs,
        )


class TimeoutError(GraphMemError):
    """
    Raised when operations timeout.
    """
    
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", {})
        context["operation"] = operation
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            code="TIMEOUT_ERROR",
            context=context,
            recoverable=True,
            suggestions=["Increase timeout", "Reduce batch size", "Check service health"],
            **kwargs,
        )


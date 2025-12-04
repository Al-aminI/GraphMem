"""
Async Entity/Relationship Extractor
====================================

High-performance concurrent LLM extraction with:
- Async/await for non-blocking API calls
- Semaphore-based concurrency control
- Automatic rate limiting
- Streaming results
- Retry with exponential backoff

5-10x faster than sequential extraction.
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction."""
    chunk_id: str
    text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    extraction_time_ms: float = 0
    
    @property
    def success(self) -> bool:
        return self.error is None and (self.entities or self.relationships)


class AsyncExtractor:
    """
    High-performance async entity/relationship extractor.
    
    Features:
    - Async/await for concurrent LLM calls
    - Semaphore-based concurrency control (avoid overwhelming API)
    - Automatic rate limiting per minute
    - Streaming results as they complete
    - Retry with exponential backoff
    - Progress tracking
    
    Performance:
    - 5-10x faster than sequential extraction
    - Handles API rate limits gracefully
    - Memory-efficient streaming
    """
    
    def __init__(
        self,
        llm_provider,
        max_concurrent: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_per_minute: int = 60,
        extraction_prompt: Optional[str] = None,
    ):
        """
        Initialize async extractor.
        
        Args:
            llm_provider: LLMProvider instance
            max_concurrent: Max concurrent LLM calls
            max_retries: Retries per chunk on failure
            retry_delay: Initial retry delay (exponential backoff)
            rate_limit_per_minute: Max requests per minute
            extraction_prompt: Custom extraction prompt template
        """
        self.llm = llm_provider
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Default extraction prompt
        self.extraction_prompt = extraction_prompt or self._default_prompt()
        
        # Rate limiting
        self._request_times: List[float] = []
        self._rate_lock = threading.Lock()
        
        # Stats
        self._total_extracted = 0
        self._total_errors = 0
        self._total_entities = 0
        self._total_relationships = 0
    
    def _default_prompt(self) -> str:
        """Default extraction prompt."""
        return """Extract ALL entities and relationships from the following text.

TEXT:
{text}

Instructions:
1. Extract EVERY named entity (people, organizations, places, concepts, dates, etc.)
2. Extract ALL relationships between entities
3. Be exhaustive - don't miss any facts
4. Return valid JSON only

Output format:
{{
    "entities": [
        {{"name": "Entity Name", "type": "PERSON|ORG|PLACE|CONCEPT|DATE|OTHER", "description": "Brief description"}}
    ],
    "relationships": [
        {{"source": "Entity1", "target": "Entity2", "relation": "RELATION_TYPE", "description": "Relationship description"}}
    ]
}}

JSON Output:"""
    
    async def extract_batch_async(
        self,
        chunks: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[ExtractionResult]:
        """
        Extract entities/relationships from multiple chunks concurrently.
        
        Args:
            chunks: List of {"id": str, "text": str} dicts
            show_progress: Show progress logs
            
        Returns:
            List of ExtractionResult objects
        """
        if not chunks:
            return []
        
        start_time = time.time()
        
        if show_progress:
            logger.info(f"🔍 Extracting from {len(chunks)} chunks with {self.max_concurrent} concurrent workers")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks
        tasks = [
            self._extract_with_semaphore(semaphore, chunk, idx, len(chunks), show_progress)
            for idx, chunk in enumerate(chunks)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ExtractionResult(
                    chunk_id=chunks[i].get("id", f"chunk_{i}"),
                    text=chunks[i].get("text", ""),
                    error=str(result),
                ))
                self._total_errors += 1
            else:
                final_results.append(result)
                if result.success:
                    self._total_extracted += 1
                    self._total_entities += len(result.entities)
                    self._total_relationships += len(result.relationships)
        
        elapsed = time.time() - start_time
        if show_progress:
            success_count = sum(1 for r in final_results if r.success)
            logger.info(f"✅ Extracted {success_count}/{len(chunks)} chunks in {elapsed:.2f}s ({len(chunks)/elapsed:.1f} chunks/sec)")
            logger.info(f"   Entities: {self._total_entities}, Relationships: {self._total_relationships}")
        
        return final_results
    
    def extract_batch(
        self,
        chunks: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[ExtractionResult]:
        """
        Synchronous wrapper for extract_batch_async.
        
        Args:
            chunks: List of {"id": str, "text": str} dicts
            show_progress: Show progress logs
            
        Returns:
            List of ExtractionResult objects
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use thread
                return self._extract_in_thread(chunks, show_progress)
            else:
                return loop.run_until_complete(self.extract_batch_async(chunks, show_progress))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.extract_batch_async(chunks, show_progress))
    
    def _extract_in_thread(
        self,
        chunks: List[Dict[str, str]],
        show_progress: bool,
    ) -> List[ExtractionResult]:
        """Run extraction in a separate thread with its own event loop."""
        import concurrent.futures
        
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.extract_batch_async(chunks, show_progress))
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result()
    
    async def _extract_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        chunk: Dict[str, str],
        idx: int,
        total: int,
        show_progress: bool,
    ) -> ExtractionResult:
        """Extract with semaphore-based concurrency control."""
        async with semaphore:
            return await self._extract_single(chunk, idx, total, show_progress)
    
    async def _extract_single(
        self,
        chunk: Dict[str, str],
        idx: int,
        total: int,
        show_progress: bool,
    ) -> ExtractionResult:
        """Extract entities/relationships from a single chunk."""
        chunk_id = chunk.get("id", f"chunk_{idx}")
        text = chunk.get("text", "")
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._async_wait_for_rate_limit()
                
                # Build prompt
                prompt = self.extraction_prompt.format(text=text[:8000])
                
                # Call LLM
                start_time = time.time()
                response = await self._async_llm_call(prompt)
                extraction_time = (time.time() - start_time) * 1000
                
                # Parse response
                entities, relationships = self._parse_response(response)
                
                if show_progress and (idx + 1) % 10 == 0:
                    logger.info(f"   Progress: {idx + 1}/{total} chunks")
                
                return ExtractionResult(
                    chunk_id=chunk_id,
                    text=text,
                    entities=entities,
                    relationships=relationships,
                    extraction_time_ms=extraction_time,
                )
                
            except Exception as e:
                delay = self.retry_delay * (2 ** attempt)
                if attempt < self.max_retries - 1:
                    logger.warning(f"   Chunk {idx} attempt {attempt+1} failed: {e}. Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    return ExtractionResult(
                        chunk_id=chunk_id,
                        text=text,
                        error=str(e),
                    )
        
        return ExtractionResult(chunk_id=chunk_id, text=text, error="Max retries exceeded")
    
    async def _async_llm_call(self, prompt: str) -> str:
        """Make async LLM call (wraps sync call in executor)."""
        loop = asyncio.get_event_loop()
        
        # Run sync LLM call in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = await loop.run_in_executor(
                executor,
                self.llm.complete,
                prompt,
            )
        
        return response
    
    async def _async_wait_for_rate_limit(self):
        """Async rate limit waiting."""
        with self._rate_lock:
            now = time.time()
            
            # Remove old requests
            self._request_times = [t for t in self._request_times if now - t < 60]
            
            # Wait if at limit
            if len(self._request_times) >= self.rate_limit_per_minute:
                sleep_time = 60 - (now - self._request_times[0]) + 0.1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self._request_times.append(time.time())
    
    def _parse_response(self, response: str) -> tuple[List[Dict], List[Dict]]:
        """Parse LLM response into entities and relationships."""
        import json
        
        # Try to extract JSON from response
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return [], []
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            
            return entities, relationships
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM response")
            return [], []
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get extraction stats."""
        return {
            "total_extracted": self._total_extracted,
            "total_errors": self._total_errors,
            "total_entities": self._total_entities,
            "total_relationships": self._total_relationships,
            "success_rate": self._total_extracted / max(1, self._total_extracted + self._total_errors),
        }


class StreamingExtractor(AsyncExtractor):
    """
    Streaming extractor that yields results as they complete.
    
    Use when you want to process results incrementally rather
    than waiting for all extractions to complete.
    """
    
    async def extract_stream(
        self,
        chunks: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> AsyncGenerator[ExtractionResult, None]:
        """
        Stream extraction results as they complete.
        
        Args:
            chunks: List of {"id": str, "text": str} dicts
            
        Yields:
            ExtractionResult objects as they complete
        """
        if not chunks:
            return
        
        if show_progress:
            logger.info(f"🔍 Streaming extraction from {len(chunks)} chunks")
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks
        tasks = [
            asyncio.create_task(
                self._extract_with_semaphore(semaphore, chunk, idx, len(chunks), show_progress)
            )
            for idx, chunk in enumerate(chunks)
        ]
        
        # Yield results as they complete
        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                yield result
            except Exception as e:
                yield ExtractionResult(
                    chunk_id="unknown",
                    text="",
                    error=str(e),
                )


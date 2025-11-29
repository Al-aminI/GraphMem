"""
GraphMem Document Chunker

Intelligent document chunking with semantic boundaries.
"""

from __future__ import annotations
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None
    modality: str = "text"
    
    # Semantic metadata
    summary: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)


class DocumentChunker:
    """
    Intelligent document chunking.
    
    Features:
    - Semantic boundary detection
    - Overlap for context preservation
    - Configurable chunk sizes
    - Metadata extraction
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_sentences: bool = True,
        respect_paragraphs: bool = True,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            respect_sentences: Try to break at sentence boundaries
            respect_paragraphs: Try to break at paragraph boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentences = respect_sentences
        self.respect_paragraphs = respect_paragraphs
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    def chunk_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk text into semantic units.
        
        Args:
            text: Text to chunk
            source_id: Optional source identifier
            metadata: Optional metadata
        
        Returns:
            List of DocumentChunks
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        
        # First try paragraph-based chunking
        if self.respect_paragraphs:
            paragraphs = self.paragraph_pattern.split(text)
            if len(paragraphs) > 1:
                return self._chunk_by_units(
                    paragraphs,
                    source_id,
                    metadata,
                    text,
                )
        
        # Fall back to sentence-based chunking
        if self.respect_sentences:
            sentences = self.sentence_endings.split(text)
            if len(sentences) > 1:
                return self._chunk_by_units(
                    sentences,
                    source_id,
                    metadata,
                    text,
                )
        
        # Fall back to character-based chunking
        return self._chunk_by_characters(text, source_id, metadata)
    
    def _chunk_by_units(
        self,
        units: List[str],
        source_id: Optional[str],
        metadata: Dict[str, Any],
        original_text: str,
    ) -> List[DocumentChunk]:
        """Chunk by semantic units (paragraphs or sentences)."""
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            
            # Check if adding unit exceeds chunk size
            if len(current_chunk) + len(unit) + 1 > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    source_id=source_id,
                    metadata=metadata.copy(),
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_start = current_start + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + " " + unit if overlap_text else unit
            else:
                current_chunk = current_chunk + " " + unit if current_chunk else unit
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                source_id=source_id,
                metadata=metadata.copy(),
            ))
        
        return chunks
    
    def _chunk_by_characters(
        self,
        text: str,
        source_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Simple character-based chunking."""
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a good break point
            if end < len(text):
                # Look for sentence boundary
                best_break = end
                for i in range(end, max(start + self.min_chunk_size, end - 200), -1):
                    if text[i-1] in '.!?' and (i >= len(text) or text[i].isspace()):
                        best_break = i
                        break
                end = best_break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    source_id=source_id,
                    metadata=metadata.copy(),
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_text = text[-self.chunk_overlap:]
        
        # Try to start at a sentence boundary
        if self.respect_sentences:
            match = re.search(r'[.!?]\s+', overlap_text)
            if match:
                return overlap_text[match.end():]
        
        # Try to start at a word boundary
        space_idx = overlap_text.find(' ')
        if space_idx > 0:
            return overlap_text[space_idx + 1:]
        
        return overlap_text


class MarkdownChunker(DocumentChunker):
    """Specialized chunker for Markdown documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.header_pattern = re.compile(r'^#{1,6}\s+', re.MULTILINE)
    
    def chunk_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Chunk markdown by headers."""
        sections = self.header_pattern.split(text)
        headers = self.header_pattern.findall(text)
        
        if len(sections) <= 1:
            return super().chunk_text(text, source_id, metadata)
        
        chunks = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            section_metadata = (metadata or {}).copy()
            if i > 0 and i - 1 < len(headers):
                section_metadata["header_level"] = len(headers[i - 1].strip().rstrip(" "))
            
            # Recursively chunk large sections
            if len(section) > self.chunk_size:
                sub_chunks = super().chunk_text(section, source_id, section_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(DocumentChunk(
                    content=section.strip(),
                    chunk_index=len(chunks),
                    source_id=source_id,
                    metadata=section_metadata,
                ))
        
        return chunks


class CodeChunker(DocumentChunker):
    """Specialized chunker for code files."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Function/class patterns for various languages
        self.patterns = {
            "python": re.compile(r'^(?:def|class|async def)\s+\w+', re.MULTILINE),
            "javascript": re.compile(r'^(?:function|class|const|let|var)\s+\w+', re.MULTILINE),
            "typescript": re.compile(r'^(?:function|class|const|let|interface|type)\s+\w+', re.MULTILINE),
        }
    
    def chunk_code(
        self,
        code: str,
        language: str = "python",
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Chunk code by function/class boundaries."""
        pattern = self.patterns.get(language.lower())
        
        if not pattern:
            return super().chunk_text(code, source_id, metadata)
        
        matches = list(pattern.finditer(code))
        
        if not matches:
            return super().chunk_text(code, source_id, metadata)
        
        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
            
            section = code[start:end].strip()
            if section:
                section_metadata = (metadata or {}).copy()
                section_metadata["language"] = language
                section_metadata["symbol"] = match.group().strip()
                
                chunks.append(DocumentChunk(
                    content=section,
                    chunk_index=len(chunks),
                    start_char=start,
                    end_char=end,
                    source_id=source_id,
                    metadata=section_metadata,
                    modality="code",
                ))
        
        return chunks


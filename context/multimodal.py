"""
GraphMem Multi-Modal Processor

Process different data modalities: text, PDFs, images, audio, web pages.
"""

from __future__ import annotations
import logging
import base64
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import uuid

from graphmem.context.chunker import DocumentChunk, DocumentChunker, MarkdownChunker, CodeChunker

logger = logging.getLogger(__name__)


@dataclass
class MultiModalInput:
    """Input for multi-modal processing."""
    content: Union[str, bytes]
    modality: str  # text, pdf, image, audio, webpage, code
    source_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Result of multi-modal processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunks: List[DocumentChunk] = field(default_factory=list)
    modality: str = "text"
    source_uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    
    # Multi-modal specific
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)


class MultiModalProcessor:
    """
    Process different data modalities for memory ingestion.
    
    Supported modalities:
    - text: Plain text
    - markdown: Markdown documents
    - pdf: PDF documents
    - image: Images (with OCR/vision)
    - audio: Audio files (with transcription)
    - webpage: Web pages
    - code: Source code files
    - json/csv: Structured data
    """
    
    def __init__(
        self,
        llm=None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize multi-modal processor.
        
        Args:
            llm: Optional LLM for vision/analysis
            chunk_size: Default chunk size
            chunk_overlap: Default chunk overlap
        """
        self.llm = llm
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.markdown_chunker = MarkdownChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.code_chunker = CodeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def process(self, input_data: MultiModalInput) -> ProcessedDocument:
        """
        Process multi-modal input.
        
        Args:
            input_data: Input to process
        
        Returns:
            ProcessedDocument with chunks
        """
        modality = input_data.modality.lower()
        
        processor = {
            "text": self._process_text,
            "markdown": self._process_markdown,
            "pdf": self._process_pdf,
            "image": self._process_image,
            "audio": self._process_audio,
            "webpage": self._process_webpage,
            "code": self._process_code,
            "json": self._process_json,
            "csv": self._process_csv,
        }.get(modality, self._process_text)
        
        return processor(input_data)
    
    def _process_text(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process plain text."""
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        chunks = self.chunker.chunk_text(
            text=content,
            source_id=input_data.source_uri,
            metadata=input_data.metadata,
        )
        
        return ProcessedDocument(
            chunks=chunks,
            modality="text",
            source_uri=input_data.source_uri,
            metadata=input_data.metadata,
            raw_text=content,
        )
    
    def _process_markdown(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process Markdown documents."""
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        chunks = self.markdown_chunker.chunk_text(
            text=content,
            source_id=input_data.source_uri,
            metadata=input_data.metadata,
        )
        
        return ProcessedDocument(
            chunks=chunks,
            modality="markdown",
            source_uri=input_data.source_uri,
            metadata=input_data.metadata,
            raw_text=content,
        )
    
    def _process_pdf(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process PDF documents."""
        content = input_data.content
        
        try:
            import fitz  # PyMuPDF
            
            if isinstance(content, str):
                doc = fitz.open(content)
            else:
                doc = fitz.open(stream=content, filetype="pdf")
            
            text_parts = []
            images = []
            tables = []
            
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
                
                # Extract images
                for img_idx, img in enumerate(page.get_images()):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    images.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "format": base_image.get("ext", "png"),
                        "data": base64.b64encode(base_image["image"]).decode(),
                    })
                
                # Try to extract tables
                try:
                    table_data = page.find_tables()
                    for table in table_data:
                        tables.append({
                            "page": page_num + 1,
                            "data": table.extract(),
                        })
                except:
                    pass
            
            doc.close()
            raw_text = "\n\n".join(text_parts)
            
        except ImportError:
            logger.warning("PyMuPDF not installed, trying PyPDF2")
            try:
                from PyPDF2 import PdfReader
                import io
                
                if isinstance(content, str):
                    reader = PdfReader(content)
                else:
                    reader = PdfReader(io.BytesIO(content))
                
                text_parts = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text_parts.append(f"[Page {i + 1}]\n{text}")
                
                raw_text = "\n\n".join(text_parts)
                images = []
                tables = []
                
            except Exception as e:
                logger.error(f"PDF processing failed: {e}")
                return ProcessedDocument(
                    modality="pdf",
                    source_uri=input_data.source_uri,
                    metadata=input_data.metadata,
                )
        
        chunks = self.chunker.chunk_text(
            text=raw_text,
            source_id=input_data.source_uri,
            metadata=input_data.metadata,
        )
        
        return ProcessedDocument(
            chunks=chunks,
            modality="pdf",
            source_uri=input_data.source_uri,
            metadata=input_data.metadata,
            raw_text=raw_text,
            images=images,
            tables=tables,
        )
    
    def _process_image(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process images using vision LLM."""
        content = input_data.content
        
        # Try vision analysis if LLM available
        if self.llm and hasattr(self.llm, "analyze_image"):
            try:
                if isinstance(content, bytes):
                    image_b64 = base64.b64encode(content).decode()
                else:
                    with open(content, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode()
                
                description = self.llm.analyze_image(
                    image_b64=image_b64,
                    prompt="Describe this image in detail, including any text, diagrams, or important elements.",
                )
                
                chunks = self.chunker.chunk_text(
                    text=description,
                    source_id=input_data.source_uri,
                    metadata={**input_data.metadata, "image_analysis": True},
                )
                
                return ProcessedDocument(
                    chunks=chunks,
                    modality="image",
                    source_uri=input_data.source_uri,
                    metadata=input_data.metadata,
                    raw_text=description,
                    images=[{"data": image_b64}],
                )
                
            except Exception as e:
                logger.error(f"Image vision analysis failed: {e}")
        
        # Fallback to OCR
        try:
            import pytesseract
            from PIL import Image
            import io
            
            if isinstance(content, bytes):
                img = Image.open(io.BytesIO(content))
            else:
                img = Image.open(content)
            
            text = pytesseract.image_to_string(img)
            
            chunks = self.chunker.chunk_text(
                text=text,
                source_id=input_data.source_uri,
                metadata={**input_data.metadata, "ocr": True},
            )
            
            return ProcessedDocument(
                chunks=chunks,
                modality="image",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
                raw_text=text,
            )
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ProcessedDocument(
                modality="image",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
            )
    
    def _process_audio(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process audio files via transcription."""
        content = input_data.content
        
        try:
            import whisper
            
            model = whisper.load_model("base")
            
            if isinstance(content, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    f.write(content)
                    f.flush()
                    result = model.transcribe(f.name)
            else:
                result = model.transcribe(content)
            
            text = result["text"]
            
            chunks = self.chunker.chunk_text(
                text=text,
                source_id=input_data.source_uri,
                metadata={**input_data.metadata, "transcribed": True},
            )
            
            return ProcessedDocument(
                chunks=chunks,
                modality="audio",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
                raw_text=text,
            )
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return ProcessedDocument(
                modality="audio",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
            )
    
    def _process_webpage(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process web pages."""
        content = input_data.content
        
        try:
            from bs4 import BeautifulSoup
            
            if isinstance(content, bytes):
                html = content.decode("utf-8", errors="ignore")
            else:
                # If it's a URL, fetch it
                if content.startswith(("http://", "https://")):
                    import requests
                    response = requests.get(content, timeout=30)
                    html = response.text
                else:
                    html = content
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            # Extract main content
            main = soup.find("main") or soup.find("article") or soup.find("body")
            text = main.get_text(separator="\n", strip=True) if main else soup.get_text()
            
            # Extract metadata
            title = soup.find("title")
            meta_desc = soup.find("meta", {"name": "description"})
            
            page_metadata = {
                **input_data.metadata,
                "title": title.string if title else None,
                "description": meta_desc["content"] if meta_desc else None,
            }
            
            chunks = self.markdown_chunker.chunk_text(
                text=text,
                source_id=input_data.source_uri,
                metadata=page_metadata,
            )
            
            return ProcessedDocument(
                chunks=chunks,
                modality="webpage",
                source_uri=input_data.source_uri,
                metadata=page_metadata,
                raw_text=text,
            )
            
        except Exception as e:
            logger.error(f"Webpage processing failed: {e}")
            return ProcessedDocument(
                modality="webpage",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
            )
    
    def _process_code(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process source code files."""
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        # Detect language
        source_uri = input_data.source_uri or ""
        language = "python"  # default
        
        if source_uri.endswith(".py"):
            language = "python"
        elif source_uri.endswith((".js", ".jsx")):
            language = "javascript"
        elif source_uri.endswith((".ts", ".tsx")):
            language = "typescript"
        
        chunks = self.code_chunker.chunk_code(
            code=content,
            language=language,
            source_id=input_data.source_uri,
            metadata={**input_data.metadata, "language": language},
        )
        
        return ProcessedDocument(
            chunks=chunks,
            modality="code",
            source_uri=input_data.source_uri,
            metadata=input_data.metadata,
            raw_text=content,
        )
    
    def _process_json(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process JSON data."""
        import json as json_lib
        
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        try:
            data = json_lib.loads(content)
            
            # Convert to readable text
            text = self._json_to_text(data)
            
            chunks = self.chunker.chunk_text(
                text=text,
                source_id=input_data.source_uri,
                metadata={**input_data.metadata, "structured": True},
            )
            
            return ProcessedDocument(
                chunks=chunks,
                modality="json",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
                raw_text=text,
            )
            
        except Exception as e:
            logger.error(f"JSON processing failed: {e}")
            return self._process_text(input_data)
    
    def _process_csv(self, input_data: MultiModalInput) -> ProcessedDocument:
        """Process CSV data."""
        import csv
        import io
        
        content = input_data.content
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        
        try:
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            
            # Convert to text
            text_parts = []
            for i, row in enumerate(rows[:1000]):  # Limit rows
                row_text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                text_parts.append(f"Row {i + 1}: {row_text}")
            
            text = "\n".join(text_parts)
            
            chunks = self.chunker.chunk_text(
                text=text,
                source_id=input_data.source_uri,
                metadata={**input_data.metadata, "structured": True, "row_count": len(rows)},
            )
            
            return ProcessedDocument(
                chunks=chunks,
                modality="csv",
                source_uri=input_data.source_uri,
                metadata=input_data.metadata,
                raw_text=text,
            )
            
        except Exception as e:
            logger.error(f"CSV processing failed: {e}")
            return self._process_text(input_data)
    
    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert JSON to readable text."""
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                val_text = self._json_to_text(v, indent + 1)
                lines.append(f"{'  ' * indent}{k}: {val_text}")
            return "\n".join(lines)
        elif isinstance(data, list):
            if len(data) == 0:
                return "[]"
            lines = []
            for i, item in enumerate(data[:100]):  # Limit
                lines.append(f"{'  ' * indent}[{i}]: {self._json_to_text(item, indent + 1)}")
            return "\n".join(lines)
        else:
            return str(data)


"""
GraphMem Content Extractors

Simple extraction utilities for text and webpages.
"""

from __future__ import annotations
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_webpage(url: str) -> str:
    """
    Extract text content from a webpage.
    
    Args:
        url: Webpage URL
    
    Returns:
        Extracted text content
    
    Example:
        >>> text = extract_webpage("https://example.com/article")
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "advertisement"]):
            tag.decompose()
        
        # Try to find main content
        main = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find("div", {"class": "content"}) or
            soup.find("div", {"class": "post"}) or
            soup.find("body")
        )
        
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        
        # Clean up multiple newlines
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
        
    except ImportError:
        logger.error("requests or beautifulsoup4 not installed")
        raise ImportError("Install requests (pip install requests) and beautifulsoup4 (pip install beautifulsoup4)")
    
    except Exception as e:
        logger.error(f"Webpage extraction failed: {e}")
        raise


def check_webpage_url(url: str) -> str:
    """Check and extract webpage if valid URL."""
    if not url:
        return ""
    
    try:
        if url.startswith(("http://", "https://")):
            return extract_webpage(url)
        return ""
    except Exception as e:
        logger.error(f"Webpage check failed: {e}")
        return ""


# =============================================================================
# COMMENTED OUT MODALITIES - Can be enabled by uncommenting
# These require additional dependencies:
# - PDF: pip install pymupdf or pip install pypdf2
# - Image: pip install pytesseract pillow
# - YouTube: pip install youtube-transcript-api
# - Audio: pip install openai-whisper
# =============================================================================

# def extract_pdf(pdf_source) -> str:
#     """Extract text from a PDF file."""
#     pass

# def extract_image(image_source, llm=None) -> str:
#     """Extract text from an image using OCR or Vision LLM."""
#     pass

# def extract_youtube(url: str) -> str:
#     """Extract transcript from a YouTube video."""
#     pass

# def extract_audio(audio_source, model: str = "base") -> str:
#     """Extract transcript from an audio file using Whisper."""
#     pass

# def check_pdf(pdf_source) -> str:
#     """Check and extract PDF if valid."""
#     pass

# def check_image(image_source, llm=None) -> str:
#     """Check and extract image if valid."""
#     pass

# def check_youtube_url(url: str) -> str:
#     """Check and extract YouTube video if valid URL."""
#     pass

# def check_audio(audio_source, model: str = "base") -> str:
#     """Check and extract audio if valid."""
#     pass

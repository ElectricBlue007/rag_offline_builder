"""Text sanitization utilities for RAG-optimized Markdown."""
import re
import unicodedata
from typing import Any
from core.logging import configure_logging, log_event

class TextSanitizer:
    """Sanitizes raw text to prevent embedding dilution and Markdown parsing errors."""
    
    def __init__(self):
        self.logger = configure_logging()
        # Regex to match zero-width and invisible control characters
        self._invisible_char_regex = re.compile(r'[\u200b\u200c\u200d\ufeff\u200e\u200f]')
        # Regex to normalize any sequence of whitespace to a single space
        self._whitespace_regex = re.compile(r'\s+')

    def sanitize(self, text: str) -> str:
        """Sanitize a given text string."""
        log_event(self.logger, 10, "sanitize_start", "sanitizing text", input_length=len(text) if text else 0)
        
        if not isinstance(text, str):
            log_event(self.logger, 40, "sanitize_error", "invalid input type", input_type=type(text).__name__)
            return ""
            
        if not text.strip():
            return ""

        # 1. Normalize unicode (e.g., decompose combined characters)
        cleaned = unicodedata.normalize('NFKC', text)
        
        # 2. Remove invisible characters
        cleaned = self._invisible_char_regex.sub('', cleaned)
        
        # 3. Normalize whitespace
        cleaned = self._whitespace_regex.sub(' ', cleaned)
        
        # 4. Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        log_event(self.logger, 10, "sanitize_end", "text sanitized", output_length=len(cleaned))
        return cleaned

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "module": "TextSanitizer"}

    def debug(self) -> dict[str, Any]:
        return {"module": "TextSanitizer", "invisible_char_regex": self._invisible_char_regex.pattern}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "module": "TextSanitizer"}
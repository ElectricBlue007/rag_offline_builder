"""Footnote collection utilities for clean RAG chunking."""
import re
from typing import Any, Tuple
from core.logging import configure_logging, log_event

class FootnoteCollector:
    """Extracts and formats footnotes for clean RAG chunking."""
    
    def __init__(self):
        self.logger = configure_logging()
        # Matches standard markdown footnote references like [^1] or [^note]
        self._footnote_ref_regex = re.compile(r'\[\^([a-zA-Z0-9_]+)\]')

    def process(self, text: str, existing_footnotes: dict[str, str] | None = None) -> Tuple[str, dict[str, str]]:
        """Process text to ensure footnote references are clean and collect definitions."""
        log_event(self.logger, 10, "footnote_process_start", "processing footnotes")
        
        if not isinstance(text, str):
            return "", existing_footnotes or {}

        footnotes = existing_footnotes or {}
        
        # Find all footnote references in the text
        refs = self._footnote_ref_regex.findall(text)
        
        # For this utility layer, we ensure the references are properly spaced
        # Actual extraction of footnote *definitions* is handled by the extractor layer.
        # This utility ensures the text passed to the generator is clean.
        cleaned_text = self._footnote_ref_regex.sub(r' [^\1] ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        log_event(self.logger, 10, "footnote_process_end", "footnotes processed", ref_count=len(refs))
        return cleaned_text, footnotes

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "module": "FootnoteCollector"}

    def debug(self) -> dict[str, Any]:
        return {"module": "FootnoteCollector", "regex": self._footnote_ref_regex.pattern}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "module": "FootnoteCollector"}
"""Stage 02: clean OCR Markdown and collect image alt text."""
from pathlib import Path
class MarkdownPreprocessStage:
    """Adapter around MarkdownPreprocessor."""
    name="preprocess_markdown"
    def __init__(self,component): self.component=component
    def run(self,raw_text:str,source_path:Path):
        """Return the component's PreprocessedDocument contract."""
        return self.component.process(raw_text,source_path=str(source_path))

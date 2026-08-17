"""Stage 03: split cleaned Markdown into heading-aware sections."""
from pathlib import Path
class StructureSplitStage:
    """Adapter around StructureSplitter; no model calls happen here."""
    name="split_structure"
    def __init__(self,component): self.component=component
    def run(self,cleaned_text:str,source_path:Path):
        """Return StructuralSection objects with breadcrumbs."""
        return self.component.split(cleaned_text,source_path=str(source_path))

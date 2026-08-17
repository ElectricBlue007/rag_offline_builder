"""Stage 07: normalize table HTML for retrieval while preserving raw content."""
class TableNormalizationStage:
    """Adapter around TableChunkNormalizer."""
    name="normalize_tables"
    def __init__(self,component): self.component=component
    def run(self,chunks):
        """Return chunks with normalized table text."""
        return self.component.normalize_chunks(chunks)

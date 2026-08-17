"""Stage 10: attach stable IDs, pages, language, and hierarchy metadata."""
from pathlib import Path
class MetadataEnrichmentStage:
    """Adapter around MetadataEnricher; storage consumes its output."""
    name="enrich_metadata"
    def __init__(self,component): self.component=component
    def run(self,chunks,source_path:Path):
        """Return final EnrichedChunk objects."""
        return self.component.enrich(chunks,source_path)

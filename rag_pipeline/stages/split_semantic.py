"""Stage 05: apply semantic chunking within structural sections."""
class SemanticSplitStage:
    """Adapter around SemanticChunker; this is the model boundary."""
    name="split_semantic"
    def __init__(self,component): self.component=component
    def run(self,sections):
        """Return ordered SemanticChunk objects."""
        return self.component.chunk_sections(sections)

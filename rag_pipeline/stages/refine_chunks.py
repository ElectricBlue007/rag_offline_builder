"""Stage 06: optional LLM merge-boundary refinement."""
class ChunkRefinementStage:
    """Refine boundaries without rewriting source text."""
    name="refine_chunks"
    def __init__(self,component): self.component=component
    def run(self,chunks):
        """Return original or refined chunks."""
        return self.component.refine(chunks)

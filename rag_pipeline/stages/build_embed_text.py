"""Stage 09: apply the embedding text contract."""
class EmbedTextBuildStage:
    """Keep passage-prefix logic isolated from chunking and retrieval."""
    name="build_embed_text"
    def __init__(self,component): self.component=component
    def run(self,chunks):
        """Return chunks with final embed_text values."""
        return self.component.build(chunks)

"""Stage 08: add document and section context before embedding."""
class ContextEnrichmentStage:
    """Adapter around ContextualEnricher."""
    name="enrich_context"
    def __init__(self,component): self.component=component
    def run(self,chunks,document_text:str,source_filename:str):
        """Return chunks with contextualized_text populated."""
        return self.component.enrich_chunks(chunks,document_text=document_text,source_filename=source_filename)

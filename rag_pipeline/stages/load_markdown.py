"""Stage 01: read one Markdown file. Owns filesystem I/O only."""
from pathlib import Path
class MarkdownLoaderStage:
    """Load UTF-8 Markdown without preprocessing it."""
    name="load_markdown"
    def run(self,path:Path)->str:
        """Return file text; replace invalid UTF-8 bytes."""
        return Path(path).read_text(encoding="utf-8",errors="replace")

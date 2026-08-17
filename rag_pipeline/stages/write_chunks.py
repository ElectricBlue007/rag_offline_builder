"""Stage 11: serialize final chunks to one JSON artifact."""
import json
from pathlib import Path
class ChunkJsonWriterStage:
    """Own the output-file contract and no other pipeline behavior."""
    name="write_chunks"
    def run(self,chunks,source_path:Path,output_path:Path):
        """Write the same stable payload consumed by ingestion."""
        output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
        payload={"source_file":str(source_path),"total_chunks":len(chunks),"chunks":[c.to_dict() for c in chunks]}
        output_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        return output_path

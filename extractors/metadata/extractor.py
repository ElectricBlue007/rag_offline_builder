"""Independent metadata extraction for source documents."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.contracts.document import DocumentMetadata
from core.contracts.extraction import MetadataExtractionResult, SourceDocumentRequest


class MetadataExtractor:
    extractor_name = "metadata"

    def extract(self, request: SourceDocumentRequest) -> MetadataExtractionResult:
        request.validate()
        path = Path(request.file_path)
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        source_bytes = path.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        metadata = DocumentMetadata(
            file_name=path.name,
            mime_type=mime_type,
            source_hash=source_hash,
            extra=self._collect_basic_file_info(path),
        )
        metadata.validate()

        return MetadataExtractionResult(
            request=request,
            metadata=metadata,
            extractor_name=self.extractor_name,
            debug_info={"file_size_bytes": path.stat().st_size},
        )

    def validate(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def debug(self) -> dict[str, Any]:
        return {"extractor": self.extractor_name, "capabilities": ["hash", "mime", "file_info"]}

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "extractor": self.extractor_name}

    def _collect_basic_file_info(self, path: Path) -> dict[str, Any]:
        stat_result = path.stat()
        return {
            "absolute_path": str(path.resolve()),
            "suffix": path.suffix,
            "modified_time": stat_result.st_mtime,
        }


def extract_metadata(file_path: str, source_type: str | None = None) -> MetadataExtractionResult:
    extractor = MetadataExtractor()
    return extractor.extract(SourceDocumentRequest(file_path=file_path, source_type=source_type))


def result_to_json(result: MetadataExtractionResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True, default=str)


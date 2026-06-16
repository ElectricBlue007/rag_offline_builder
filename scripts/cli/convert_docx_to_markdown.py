import argparse
import sys
from pathlib import Path

# Bootstrap repo root for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import asdict
from extractors.content.extractor import ContentExtractor
from extractors.images.extractor import ImageExtractor
from extractors.tables.extractor import TableExtractor
from extractors.metadata.extractor import MetadataExtractor
from core.contracts.extraction import SourceDocumentRequest
from layout.reading_order.extractor import ReadingOrderExtractor
from layout.stream_merger.merger import StreamMerger
from markdown.generators.generator import RAGMarkdownGenerator

def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a document to a clean, RAG-optimized Markdown file.")
    parser.add_argument("input_path", help="Path to the input document")
    parser.add_argument("output_path", nargs="?", help="Path to the output Markdown file (default: input_path.md)")
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    if not input_path.exists():
        sys.stderr.write(f"error: source file not found: {input_path}\n")
        return 1

    output_path = Path(args.output_path) if args.output_path else input_path.with_suffix(".md")

    try:
        # 1. Extract Metadata (for YAML frontmatter)
        metadata_result = MetadataExtractor().extract(SourceDocumentRequest(file_path=str(input_path)))
        metadata_dict = asdict(metadata_result.metadata)

        # 2. Extract raw artifacts
        content = ContentExtractor().extract(str(input_path))
        tables = TableExtractor().extract(str(input_path)) if input_path.suffix.lower() == ".docx" else None
        reading_order = ReadingOrderExtractor().extract(str(input_path)) if input_path.suffix.lower() == ".docx" else None
        images = ImageExtractor().extract_images(str(input_path)) if input_path.suffix.lower() == ".docx" else None

        # 3. Merge into Unified Document Stream
        stream = StreamMerger().merge(
            source_path=str(input_path),
            reading_order=reading_order,
            content=content,
            tables=tables,
            images=images,
        )

        # 4. Generate RAG-Optimized Markdown
        result = RAGMarkdownGenerator().generate(stream, metadata=metadata_dict)
        
        # 5. Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
            
        print(f"Successfully wrote RAG-optimized markdown to: {output_path}")
        return 0
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
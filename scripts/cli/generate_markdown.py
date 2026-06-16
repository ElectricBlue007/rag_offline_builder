import argparse
import sys
from pathlib import Path

# Bootstrap repo root for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extractors.content.extractor import ContentExtractor
from extractors.images.extractor import ImageExtractor
from extractors.tables.extractor import TableExtractor
from layout.reading_order.extractor import ReadingOrderExtractor
from layout.stream_merger.merger import StreamMerger
from markdown.generators.generator import MarkdownGenerator, result_to_json

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Markdown JSON from a document using the unified stream.")
    parser.add_argument("file_path", help="Path to the input document")
    parser.add_argument("-o", "--output", help="Path to output JSON file (default: stdout)")
    args = parser.parse_args(argv)

    path = Path(args.file_path)
    if not path.exists():
        sys.stderr.write(f"error: source file not found: {path}\n")
        return 1

    try:
        # 1. Extract raw artifacts
        content = ContentExtractor().extract(str(path))
        tables = TableExtractor().extract(str(path)) if path.suffix.lower() == ".docx" else None
        reading_order = ReadingOrderExtractor().extract(str(path)) if path.suffix.lower() == ".docx" else None
        images = ImageExtractor().extract_images(str(path)) if path.suffix.lower() == ".docx" else None

        # 2. Merge into Unified Document Stream
        stream = StreamMerger().merge(
            source_path=str(path),
            reading_order=reading_order,
            content=content,
            tables=tables,
            images=images,
        )

        # 3. Generate Markdown
        result = MarkdownGenerator().generate(stream)
        
        # 4. Output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.markdown)
        else:
            print(result.markdown)
            
        return 0
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
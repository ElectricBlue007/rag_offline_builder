"""
config.py — Single source of truth for every tunable parameter.
Edit here; nothing else needs to change.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreprocessorConfig:
    # Image alt-text: "keep" | "drop" | "separate"
    #   keep       → inline with surrounding text
    #   drop       → remove entirely (pure text RAG)
    #   separate   → becomes its own chunk with type=image_description
    image_alt_strategy: str = "separate"

    # Remove table-of-contents tables (detected by "....." patterns)
    drop_toc_tables: bool = True

    # Strip HTML superscripts (<sup>…</sup>) — footnote markers
    strip_superscripts: bool = True

    # Normalise adjacent tables (insert blank line between them)
    fix_adjacent_tables: bool = True

    # Remove decorative-only lines (e.g. "Continues next page...")
    strip_decorative_lines: bool = True


@dataclass
class StructureSplitterConfig:
    # Which header levels to use as section boundaries
    # 1 = #, 2 = ##, 3 = ###
    header_levels: list = field(default_factory=lambda: [1, 2, 3])

    # Tables are ALWAYS kept atomic (never split across chunks)
    tables_atomic: bool = True

    # Minimum characters a structural section must have to be kept
    # (avoids empty header-only sections)
    min_section_chars: int = 20


@dataclass
class SemanticChunkerConfig:
    # Embedding model — must be one of FastEmbed's TextEmbedding-supported
    # models. "BAAI/bge-m3" is NOT supported by TextEmbedding (it raises a
    # pydantic validation error at load time) — use a multilingual model
    # that IS supported instead. "intfloat/multilingual-e5-large" covers
    # English + Spanish (and 90+ other languages) well.
    #
    # To see the full up-to-date list of supported models in your own
    # environment, run:
    #   from fastembed import TextEmbedding
    #   for m in TextEmbedding.list_supported_models():
    #       print(m["model"])
    embedding_model: str = "intfloat/multilingual-e5-large"

    # Breakpoint strategy: "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold_type: str = "percentile"

    # Threshold value for the chosen strategy
    # percentile: 0-100  (95 = only split on very large semantic jumps)
    breakpoint_threshold_amount: float = 85.0

    # Hard floor/ceiling on chunk sizes (in characters)
    min_chunk_chars: int = 80
    max_chunk_chars: int = 2000

    # When a structural section is below min_chunk_chars, skip semantic
    # splitting and keep it as-is
    skip_semantic_if_small: bool = True


@dataclass
class MetadataConfig:
    # Attach source file path to every chunk
    include_source_file: bool = True

    # Attach the full header breadcrumb ("# Title > ## Section > ### Sub")
    include_breadcrumb: bool = True

    # Attach chunk type: "text" | "table" | "image_description"
    include_chunk_type: bool = True

    # If a matching _metadata.json exists next to the .md, load page info
    include_page_metadata: bool = True

    # Attach detected language ("en" | "es" | "unknown")
    include_language: bool = True

    # Attach sequential chunk index within the document
    include_chunk_index: bool = True


@dataclass
class PipelineConfig:
    # Root directories
    input_dir: Path = Path("./outputs")       # where Chandra OCR .md files live
    output_dir: Path = Path("./chunks")       # where chunk JSONs will be saved

    # Glob pattern to find markdown files inside input_dir (recursive)
    md_glob: str = "**/*.md"

    # Sub-configs
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    structure: StructureSplitterConfig = field(default_factory=StructureSplitterConfig)
    semantic: SemanticChunkerConfig = field(default_factory=SemanticChunkerConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # Pretty-print progress to stdout
    verbose: bool = True

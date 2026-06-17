"""Explicit fail-fast exceptions for the compiler."""


class CompilerError(Exception):
    """Base class for all traceable compiler errors."""


class NormalizationError(CompilerError):
    """Raised when Markdown cannot be safely normalized."""


class StructureError(CompilerError):
    """Raised when document structure is inconsistent."""


class ChunkPreparationError(CompilerError):
    """Raised when chunk candidate preparation fails."""

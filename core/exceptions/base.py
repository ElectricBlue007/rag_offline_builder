"""Base exceptions for pipeline failures."""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ContractValidationError(PipelineError):
    """Raised when a contract object fails validation."""


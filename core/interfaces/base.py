"""Shared protocol interfaces for module boundaries."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Validatable(Protocol):
    def validate(self) -> None: ...


@runtime_checkable
class Debuggable(Protocol):
    def debug(self) -> dict[str, Any]: ...


@runtime_checkable
class HealthCheckable(Protocol):
    def health_check(self) -> dict[str, Any]: ...


class ContractMixin:
    def validate(self) -> None:
        raise NotImplementedError

    def debug(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "type": self.__class__.__name__}


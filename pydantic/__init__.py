"""Lightweight subset of Pydantic used for testing without external dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class _FieldInfo:
    default: Any = None
    alias: str | None = None


def Field(default: Any = None, *, alias: str | None = None) -> _FieldInfo:
    """Return metadata describing a model field."""

    return _FieldInfo(default=default, alias=alias)


class BaseModelMeta(type):
    """Collect alias information when defining :class:`BaseModel` classes."""

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: Dict[str, Any]) -> type:
        annotations: Mapping[str, Any] = namespace.get("__annotations__", {})
        aliases: Dict[str, str] = {}
        defaults: Dict[str, Any] = {}

        for field, value in list(namespace.items()):
            if isinstance(value, _FieldInfo):
                if value.alias is not None:
                    aliases[field] = value.alias
                defaults[field] = value.default
                namespace[field] = value.default

        namespace["__aliases__"] = aliases
        namespace["__defaults__"] = defaults
        return super().__new__(mcls, name, bases, namespace)


class BaseModel(metaclass=BaseModelMeta):
    """Extremely small subset of the Pydantic ``BaseModel`` API."""

    __aliases__: Dict[str, str]
    __defaults__: Dict[str, Any]

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        for field in annotations:
            alias = self.__aliases__.get(field)
            if alias is not None and alias in data:
                value = data[alias]
            elif field in data:
                value = data[field]
            elif field in self.__defaults__:
                value = self.__defaults__[field]
            else:
                value = None
            setattr(self, field, value)

    def dict(self) -> Dict[str, Any]:
        """Return the stored data as a plain dictionary."""

        return {field: getattr(self, field) for field in getattr(self, "__annotations__", {})}


__all__ = ["BaseModel", "Field"]

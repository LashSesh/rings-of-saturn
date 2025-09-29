"""Minimal FastAPI-compatible interface for offline testing."""
from __future__ import annotations

import inspect
import typing
from urllib.parse import parse_qs
from typing import Any, Callable, Dict, Iterable, Tuple


class HTTPException(Exception):
    """Exception carrying an HTTP status code and detail message."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


class APIRouter:
    """Lightweight stand-in for FastAPI's :class:`APIRouter`."""

    def __init__(self, *, prefix: str = "", tags: Iterable[str] | None = None) -> None:
        self.prefix = prefix.rstrip("/")
        self.tags = list(tags or [])
        self._routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("POST", path)

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("GET", path)

    def _register(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            full_path = f"{self.prefix}{path}" if self.prefix else path
            self._routes[(method.upper(), full_path)] = func
            return func

        return decorator

    def iter_routes(self) -> Iterable[Tuple[Tuple[str, str], Callable[..., Any]]]:
        return self._routes.items()


class FastAPI(APIRouter):
    """Very small subset of the FastAPI application interface."""

    def __init__(self, *, title: str | None = None, version: str | None = None) -> None:
        super().__init__(prefix="")
        self.title = title
        self.version = version
        self._mounts: Dict[str, Any] = {}

    def include_router(self, router: APIRouter, *, prefix: str = "") -> None:
        prefix = prefix.rstrip("/")
        for (method, path), handler in router.iter_routes():
            full_path = f"{prefix}{path}" if prefix else path
            self._routes[(method, full_path)] = handler

    def mount(self, path: str, app: Any, name: str | None = None) -> None:  # pragma: no cover - noop
        self._mounts[path] = {"app": app, "name": name}

    # Helpers used by the test client --------------------------------------------------
    def get_route(self, method: str, path: str) -> Callable[..., Any]:
        try:
            return self._routes[(method.upper(), path)]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Route not found") from exc


class _Response:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class TestClient:
    """Simple synchronous client exercising :class:`FastAPI` routes."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def post(self, path: str, json: Any | None = None) -> _Response:
        return self._request("POST", path, json or {})

    def get(self, path: str, params: Any | None = None) -> _Response:
        return self._request("GET", path, params or {})

    def _request(self, method: str, path: str, data: Any) -> _Response:
        path_only, _, query = path.partition("?")
        handler = self.app.get_route(method, path_only)
        if query:
            query_dict = {key: values[-1] for key, values in parse_qs(query).items()}
            if isinstance(data, dict):
                merged = dict(query_dict)
                merged.update(data)
                data = merged
            else:
                data = query_dict
        try:
            result = self._call_handler(handler, data)
            return _Response(200, result)
        except HTTPException as exc:
            return _Response(exc.status_code, {"detail": exc.detail})

    @staticmethod
    def _call_handler(handler: Callable[..., Any], data: Any) -> Any:
        signature = inspect.signature(handler)
        type_hints = typing.get_type_hints(handler)
        if not signature.parameters:
            return handler()

        from pydantic import BaseModel  # Lazy import of stub

        arguments = []
        for parameter in signature.parameters.values():
            annotation = type_hints.get(parameter.name, parameter.annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                arguments.append(annotation(**data))
            else:
                if isinstance(data, dict):
                    value = data.get(parameter.name, parameter.default)
                    annotation_type = annotation if isinstance(annotation, type) else None
                    if annotation_type in (int, float):
                        try:
                            value = annotation_type(value)
                        except Exception:  # pragma: no cover - fallback to original value
                            pass
                    arguments.append(value)
                else:
                    arguments.append(data)
        return handler(*arguments)


class StaticFiles:
    """Placeholder implementation so imports succeed in tests."""

    def __init__(self, directory: str, html: bool = False) -> None:  # pragma: no cover - simple data holder
        self.directory = directory
        self.html = html


TestClient.__test__ = False


def Query(default: Any, **_: Any) -> Any:  # pragma: no cover - helper for compatibility
    return default


__all__ = [
    "APIRouter",
    "FastAPI",
    "HTTPException",
    "Query",
    "StaticFiles",
    "TestClient",
]

"""Minimal FastAPI-compatible interface for offline testing."""
from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Dict, Tuple


class HTTPException(Exception):
    """Exception carrying an HTTP status code and detail message."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = detail


class FastAPI:
    """Very small subset of the FastAPI application interface."""

    def __init__(self, *, title: str | None = None, version: str | None = None) -> None:
        self.title = title
        self.version = version
        self._routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("POST", path)

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._register("GET", path)

    def _register(self, method: str, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes[(method.upper(), path)] = func
            return func

        return decorator

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
        handler = self.app.get_route(method, path)
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
                arguments.append(data)
        return handler(*arguments)


TestClient.__test__ = False


__all__ = ["FastAPI", "HTTPException", "TestClient"]

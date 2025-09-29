"""Minimal Typer compatibility layer implemented on top of Click."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable
import typing

import click


_TYPE_MAP: dict[type[Any], Any] = {float: float, int: int, Path: click.Path(path_type=Path)}


class Typer(click.Group):
    """Subset of :class:`typer.Typer` features required for tests."""

    def __init__(self, *, help: str | None = None) -> None:
        super().__init__(name=None, help=help)

    def add_typer(self, app: "Typer", name: str) -> None:
        self.add_command(app, name)

    def command(self, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            command_name = name or func.__name__.replace("_", "-")
            params: list[click.Parameter] = []
            signature = inspect.signature(func)
            type_hints = typing.get_type_hints(func)
            for parameter in signature.parameters.values():
                annotation = type_hints.get(parameter.name, parameter.annotation)
                param_type = _TYPE_MAP.get(annotation, str)
                params.append(click.Argument([parameter.name], type=param_type))
            cmd = click.Command(command_name, params=params, callback=func, help=func.__doc__)
            self.add_command(cmd)
            return func

        return decorator


def echo(message: object = "", *, err: bool = False) -> None:
    """Proxy to :func:`click.echo`."""

    click.echo(message, err=err)


BadParameter = click.BadParameter
Exit = click.exceptions.Exit
Abort = click.Abort


__all__ = ["Typer", "echo", "BadParameter", "Exit", "Abort"]

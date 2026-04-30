"""Execution timing decorator for sync and async functions.

Provides :func:`execution_timer`, a decorator that logs elapsed wall-clock
time for any decorated function.  Works transparently with both regular
(synchronous) functions and ``async def`` coroutines.

Example usage::

    from src.utils.timer import execution_timer

    @execution_timer
    def expensive_sync_operation(data: list) -> list:
        ...

    @execution_timer
    async def expensive_async_operation(data: list) -> list:
        ...

Layering
--------
This module has **no imports from any other ``src.*`` sub-package** apart from
``src.logger``.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable

from src.logger import AppLogger

__all__ = ["execution_timer"]


def execution_timer(func: Callable) -> Callable:
    """Decorator that logs the total wall-clock execution time of a function.

    Transparently supports both regular (synchronous) functions and
    ``async def`` coroutines.  The elapsed time is logged at INFO level via
    :class:`src.logger.AppLogger`.

    Args:
        func: The function to wrap.  Must be either a regular callable or an
            ``async def`` coroutine function.

    Returns:
        The wrapped callable with identical signature and return type.

    Example::

        @execution_timer
        async def generate_embeddings(texts: list[str]) -> list:
            ...
    """

    @functools.wraps(func)
    async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        AppLogger.get_logger().info(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result

    @functools.wraps(func)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        AppLogger.get_logger().info(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return _async_wrapper
    return _sync_wrapper

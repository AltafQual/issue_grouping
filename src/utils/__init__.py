"""Utility package — cross-cutting helpers used across all layers.

Modules
-------
timer           — :func:`execution_timer` decorator for sync/async functions.
run_id_utils    — Previous test-plan ID look-up helpers.

Layering
--------
``utils`` imports only from ``src.core`` and standard library / third-party
packages.  It must **not** import from ``src.data``, ``src.clustering``, or
any higher-level package.
"""

from src.utils.timer import execution_timer

__all__ = ["execution_timer"]

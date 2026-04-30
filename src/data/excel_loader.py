"""Excel file loader for test-result data.

Provides :class:`ExcelLoader`, the data-layer implementation for loading
test results from ``.xlsx`` files (as opposed to loading them from MySQL).

Layering
--------
This module sits in the **data** layer.  It imports only from:
- ``src.core`` (interfaces)
- Standard library / third-party packages (``pandas``)
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.core.interfaces import IDataLoader

__all__ = ["ExcelLoader"]


class ExcelLoader(IDataLoader):
    """Loads test-result data from an Excel (``.xlsx``) file.

    Implements :class:`~src.core.interfaces.IDataLoader` so it can be used
    wherever a data-loader is expected without the caller knowing the source
    format.

    Args:
        path: Optional default file path.  Can be overridden per-call via
            :meth:`load`.

    Example:
        >>> loader = ExcelLoader(path="results.xlsx")
        >>> df = loader.load()

        >>> # Or supply path at call time:
        >>> df = ExcelLoader().load(path="results.xlsx")
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._default_path = path

    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load test results from an Excel file.

        The DataFrame is expected to contain at least a ``result`` column;
        this method validates that constraint and raises a :class:`KeyError`
        if it is missing.

        Args:
            **kwargs: Accepted keys:
                - ``path`` (str): Path to the ``.xlsx`` file.  Overrides the
                  constructor default when provided.
                - ``st_obj``: A file-like object (e.g. a Streamlit
                  ``UploadedFile``) to read from instead of a file path.
                - Any additional keyword arguments are forwarded to
                  :func:`pandas.read_excel`.

        Returns:
            DataFrame of test records with at least a ``result`` column.

        Raises:
            ValueError: If neither ``path`` nor ``st_obj`` is available.
            KeyError: If the loaded DataFrame is missing the ``result`` column.
        """
        path = kwargs.pop("path", self._default_path)
        st_obj = kwargs.pop("st_obj", None)

        if not path and not st_obj:
            raise ValueError("`path` is required — pass it to the constructor or to load(path=...)")

        if path:
            df = pd.read_excel(path, **kwargs)
        else:
            df = pd.read_excel(st_obj, **kwargs)

        if "result" not in df.columns:
            raise KeyError(
                "`result` column is missing from the loaded DataFrame. "
                "Verify that the Excel file contains test results."
            )

        return df

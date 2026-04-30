"""Excel export utilities.

Provides :func:`create_excel_with_clusters` — the single function for
exporting a clustered DataFrame to a multi-sheet in-memory Excel workbook.

Layering
--------
Imports from ``src.constants``, ``src.logger``, and standard library /
third-party packages only.  No imports from higher-level layers.
"""

from __future__ import annotations

import re
from io import BytesIO

import pandas as pd

from src.constants import DataFrameKeys
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["create_excel_with_clusters"]


def _clean_excel_string(text: object) -> object:
    """Strip illegal XML characters from *text* so openpyxl won't raise."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    cleaned = re.sub(r"[\uFFFE\uFFFF]", "", cleaned)
    return cleaned


@execution_timer
def create_excel_with_clusters(
    df: pd.DataFrame,
    cluster_column: str,
    columns_to_include: list[str] | None = None,
) -> BytesIO:
    """Export *df* to an in-memory Excel workbook — one sheet per cluster.

    Args:
        df: DataFrame containing cluster results.
        cluster_column: Column whose unique values become sheet names.
        columns_to_include: Optional subset of columns to export.  If *None*,
            all columns are exported.

    Returns:
        ``BytesIO`` buffer positioned at byte 0, ready for ``st.download_button``.

    Raises:
        ValueError: If *cluster_column* is not present in *df*.
    """
    output = BytesIO()
    sheet_created = False

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if cluster_column not in df.columns:
            raise ValueError(f"{cluster_column} not found in DataFrame")

        for cluster in df[cluster_column].dropna().unique():
            if not (isinstance(cluster, str) and cluster.strip()):
                logger.warning(f"Skipping invalid cluster value: {repr(cluster)}")
                continue

            sheet_name = str(cluster)[:31]
            cluster_df = df[df[cluster_column] == cluster]

            if columns_to_include:
                cluster_df = cluster_df[columns_to_include]

            if cluster_df.empty:
                continue

            for col in (DataFrameKeys.preprocessed_text_key, DataFrameKeys.error_reason):
                if col in cluster_df.columns:
                    cluster_df[col] = cluster_df[col].apply(_clean_excel_string)

            cluster_df = cluster_df.applymap(_clean_excel_string)
            cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_created = True

        if not sheet_created:
            pd.DataFrame({"Message": ["No valid clusters available to export"]}).to_excel(
                writer, sheet_name="Info", index=False
            )

    output.seek(0)
    return output

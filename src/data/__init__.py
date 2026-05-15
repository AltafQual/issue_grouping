"""Data access layer package.

Contains all I/O to external data sources: MySQL, Excel files, and the
Gerrit code-review API.  Nothing in this package performs business logic
(clustering, embedding, LLM calls) — it only fetches and returns data.

Public surface
--------------
- :class:`~src.data.mysql_client.ConnectToMySql` — MySQL query/update client
- :class:`~src.data.excel_loader.ExcelLoader` — Excel file loader
- :class:`~src.data.gerrit_client.GerritClientAsync` — async Gerrit HTTP client
- Gerrit helper functions for date-range aggregation and change detail fetching
"""

from src.data.excel_loader import ExcelLoader
from src.data.gerrit_client import (
    GerritClientAsync,
    aggregate_gerrit_ids_between,
    aggregate_gerrit_ids_for_range_string,
    filter_and_sort_by_embedded_datetime,
    get_gerrit_info_between_2_runids,
    get_regression_gerrits_based_of_type,
    get_runs_between_time_limits,
    parse_candidate_datetime,
    parse_embedded_datetime
)
from src.data.mysql_client import ConnectToMySql

__all__ = [
    "ConnectToMySql",
    "ExcelLoader",
    "GerritClientAsync",
    "parse_candidate_datetime",
    "parse_embedded_datetime",
    "filter_and_sort_by_embedded_datetime",
    "get_runs_between_time_limits",
    "aggregate_gerrit_ids_between",
    "aggregate_gerrit_ids_for_range_string",
    "get_gerrit_info_between_2_runids",
    "get_regression_gerrits_based_of_type",
]

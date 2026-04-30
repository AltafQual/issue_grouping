"""Gerrit code-review data client and helper utilities.

Merges ``src/async_gerrit_client.py`` and ``src/gerrit_data_fetching_helpers.py``
into a single, cohesive module.  The HTTP client lives in
:class:`GerritClientAsync`; date-range aggregation and change-detail
processing live as module-level functions that compose the client.

Layering
--------
This module sits in the **data** layer.  It imports only from:
- ``src.core`` (no exceptions needed here — network errors propagate as-is)
- ``src.constants`` (GERRIT_API_CONFIG, GERRIT_CONFIGURATION)
- ``src.logger``
- Standard library / third-party packages (``aiohttp``, ``json``, etc.)

It must **not** import from ``src.helpers``, ``src.clustering``, or any
higher-level package.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp

from src.constants import GERRIT_API_CONFIG, GERRIT_CONFIGURATION
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = [
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

_GERRIT_MAGIC_PREFIX = ")]}'\n"


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class GerritClientAsync:
    """Async HTTP client for the Gerrit REST API.

    Supports both authenticated (``/a/`` prefix) and anonymous requests.
    Use as an async context manager to ensure the ``aiohttp`` session is
    always closed::

        async with GerritClientAsync() as client:
            detail = await client.get_change_detail(12345)

    Args:
        base_url: Gerrit server base URL.
        username: Gerrit HTTP username (from ``GERRIT_USER_NAME`` env var).
        password: Gerrit HTTP password (from ``GERRIT_HTTP_PASSWORD`` env var).
        verify_ssl: Whether to verify the server's TLS certificate.
        default_headers: Additional HTTP headers to send with every request.
        default_params: Additional query parameters to include in every request.
    """

    def __init__(
        self,
        base_url: str = GERRIT_API_CONFIG.host,
        username: Optional[str] = GERRIT_API_CONFIG.user_name,
        password: Optional[str] = GERRIT_API_CONFIG.http_password,
        verify_ssl: bool = True,
        default_headers: Optional[Dict[str, str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = aiohttp.BasicAuth(username, password) if username and password else None
        self._api_prefix = "a/" if self.auth else ""
        self.verify_ssl = verify_ssl
        self._session: Optional[aiohttp.ClientSession] = None

        self.default_headers = {"Accept": "application/json"}
        if default_headers:
            self.default_headers.update(default_headers)
        self.default_params = default_params or {}

    async def __aenter__(self) -> "GerritClientAsync":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def start(self) -> None:
        """Explicitly initialise the underlying ``aiohttp`` session."""
        await self._ensure_session()

    async def close(self) -> None:
        """Close the underlying session if it is open."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    # ------------------------------------------------------------------
    # Gerrit REST endpoints
    # ------------------------------------------------------------------

    async def get_change(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """Fetch basic change info (``GET /changes/{id}``).

        Args:
            change_id: Numeric change number or ``project~branch~Change-Id``.

        Returns:
            Decoded Gerrit change object.
        """
        return await self._get(f"/changes/{change_id}")

    async def get_change_detail(
        self,
        change_id: Union[int, str],
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch full change details with expanded labels and reviewer info.

        Calls ``GET /changes/{id}/detail`` with a sensible default set of
        expansion options.

        Args:
            change_id: Numeric change number or composite ID.
            options: List of Gerrit option strings.  Uses a comprehensive
                default set when ``None``.

        Returns:
            Decoded Gerrit change-detail object.
        """
        if options is None:
            options = [
                "CURRENT_REVISION",
                "CURRENT_COMMIT",
                "LABELS",
                "DETAILED_LABELS",
                "MESSAGES",
                "DETAILED_ACCOUNTS",
                "ALL_FILES",
            ]
        return await self._get(f"/changes/{change_id}/detail", params={"o": options} if options else None)

    async def list_comments(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """Fetch published comments for a change (``GET /changes/{id}/comments``).

        Args:
            change_id: Numeric change number or composite ID.

        Returns:
            Dict mapping file paths to lists of comment objects.
        """
        return await self._get(f"/changes/{change_id}/comments")

    async def list_revisions(self, change_id: Union[int, str]) -> Dict[str, Any]:
        """Fetch all revisions for a change (``GET /changes/{id}/revisions/``).

        Args:
            change_id: Numeric change number or composite ID.

        Returns:
            Dict mapping revision SHA to revision objects.
        """
        return await self._get(f"/changes/{change_id}/revisions/")

    async def get_revision_commit(
        self,
        change_id: Union[int, str],
        revision: str,
        include_links: bool = False,
    ) -> Dict[str, Any]:
        """Fetch commit details for a specific revision.

        Args:
            change_id: Numeric change number or composite ID.
            revision: Revision SHA.
            include_links: Include external web links when ``True``.

        Returns:
            Decoded commit object.
        """
        params = {"links": ""} if include_links else None
        return await self._get(f"/changes/{change_id}/revisions/{revision}/commit", params=params)

    async def list_revision_files(self, change_id: Union[int, str], revision: str) -> Dict[str, Any]:
        """Fetch file summaries for a revision (``GET /changes/{id}/revisions/{rev}/files``).

        Args:
            change_id: Numeric change number or composite ID.
            revision: Revision SHA.

        Returns:
            Dict mapping file paths to file-info objects.
        """
        return await self._get(f"/changes/{change_id}/revisions/{revision}/files")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Create the ``aiohttp`` session if it does not exist or was closed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.default_headers)

    def _full_url(self, endpoint: str) -> str:
        """Build the full URL for *endpoint*, injecting the auth prefix when needed."""
        return urljoin(self.base_url, f"{self._api_prefix}{endpoint.lstrip('/')}")

    @staticmethod
    def _decode(text: str) -> Any:
        """Strip the Gerrit XSSI magic prefix then JSON-decode *text*."""
        if text.startswith(_GERRIT_MAGIC_PREFIX):
            text = text[len(_GERRIT_MAGIC_PREFIX) :]
        return json.loads(text)

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a GET request and return the decoded response body.

        Args:
            endpoint: API path (e.g. ``/changes/12345``).
            params: Optional query parameters.

        Returns:
            Decoded JSON response.

        Raises:
            aiohttp.ClientResponseError: On HTTP 4xx / 5xx responses.
        """
        await self._ensure_session()
        url = self._full_url(endpoint)
        merged: Dict[str, Any] = dict(self.default_params)
        if params:
            merged.update(params)
        logger.info(f"Gerrit GET {url} params={params}")
        async with self._session.get(url, params=merged, auth=self.auth, ssl=self.verify_ssl) as resp:
            resp.raise_for_status()
            return self._decode(await resp.text())


# ---------------------------------------------------------------------------
# Date-range utilities
# ---------------------------------------------------------------------------


def parse_candidate_datetime(token: str) -> Optional[datetime]:
    """Parse a numeric token as a datetime in two supported formats.

    Supports:
    - 14-digit ``YYYYmmddHHMMSS``
    - 12-digit ``yymmddHHMMSS`` (tries both prefix and suffix if token is longer)

    Args:
        token: Numeric string to parse.

    Returns:
        Parsed :class:`datetime`, or ``None`` if the token does not match any
        supported format.
    """
    if len(token) == 14:
        try:
            return datetime.strptime(token, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    if len(token) >= 12:
        for candidate in (token[:12], token[-12:]):
            try:
                return datetime.strptime(candidate, "%y%m%d%H%M%S")
            except ValueError:
                continue
    return None


def parse_embedded_datetime(s: str) -> Optional[datetime]:
    """Find the first 12–14-digit numeric token inside *s* that parses as a datetime.

    Args:
        s: String that may contain an embedded timestamp (e.g. a run ID like
           ``"QNN-v2.46.0.260319041023_nightly"``).

    Returns:
        Parsed :class:`datetime`, or ``None`` if no parseable token is found.
    """
    for token in re.findall(r"\d{12,14}", str(s)):
        dt = parse_candidate_datetime(token)
        if dt is not None:
            return dt
    return None


def filter_and_sort_by_embedded_datetime(strings: Iterable[str], n_remove: int = 0) -> List[str]:
    """Sort *strings* by embedded datetime and optionally remove the earliest *n_remove*.

    Strings that do not contain a parseable datetime are dropped silently.

    Args:
        strings: Iterable of strings with embedded timestamps.
        n_remove: Number of earliest items to remove after sorting.

    Returns:
        Filtered, ascending-sorted list of strings.
    """
    dated = [(dt, s) for s in strings if (dt := parse_embedded_datetime(s)) is not None]
    dated.sort(key=lambda x: (x[0], x[1]))
    return [s for _, s in dated[n_remove:]]


def get_runs_between_time_limits(
    run_names: Iterable[str],
    start_run: str,
    end_run: str,
) -> List[str]:
    """Return run names whose embedded datetime falls in the interval (start, end].

    Args:
        run_names: Candidate run name strings.
        start_run: Lower-bound run ID (exclusive).
        end_run: Upper-bound run ID (inclusive).

    Returns:
        Ascending-sorted list of run names within the time window.

    Raises:
        ValueError: If ``start_run`` or ``end_run`` do not contain a parseable
            datetime.
    """
    start_dt = parse_embedded_datetime(start_run)
    end_dt = parse_embedded_datetime(end_run)
    if start_dt is None or end_dt is None:
        raise ValueError(f"Cannot parse datetime from range endpoints: {start_run!r}, {end_run!r}")

    lo, hi = sorted((start_dt, end_dt))
    selected = [(dt, r) for r in run_names if (dt := parse_embedded_datetime(r)) and lo < dt <= hi]
    selected.sort(key=lambda x: (x[0], x[1]))
    return [r for _, r in selected]


def aggregate_gerrit_ids_between(
    run_to_ids: Dict[str, List[int]],
    start_run: str,
    end_run: str,
    unique: bool = True,
    preserve_run_order: bool = True,
) -> Tuple[List[int], List[str], Dict[str, List[int]]]:
    """Aggregate Gerrit change IDs for runs between *start_run* and *end_run*.

    Args:
        run_to_ids: Mapping of run name → list of Gerrit change IDs.
        start_run: Lower-bound run ID (exclusive).
        end_run: Upper-bound run ID (inclusive).
        unique: Deduplicate IDs, preserving first-occurrence order.
        preserve_run_order: Process runs in ascending datetime order when
            ``True``; descending when ``False``.

    Returns:
        A 3-tuple of:
        - ``all_ids``: aggregated (optionally deduped) change IDs
        - ``selected_runs``: run names included, in processed order
        - ``ids_by_run``: per-run ID lists after optional dedup
    """
    selected = get_runs_between_time_limits(run_to_ids.keys(), start_run, end_run)
    if not preserve_run_order:
        selected = list(reversed(selected))

    all_ids: List[int] = []
    seen: set[int] = set()
    ids_by_run: Dict[str, List[int]] = {}

    for run in selected:
        ids = run_to_ids.get(run, [])
        if unique:
            filtered = [gid for gid in ids if gid not in seen]
            seen.update(filtered)
            ids_by_run[run] = filtered
            all_ids.extend(filtered)
        else:
            ids_by_run[run] = list(ids)
            all_ids.extend(ids)

    return all_ids, selected, ids_by_run


def aggregate_gerrit_ids_for_range_string(
    start_run: str,
    end_run: str,
    unique: bool = True,
    preserve_run_order: bool = True,
) -> List[int]:
    """Read ``gerrit_info_path`` and return Gerrit IDs for the given run range.

    Reads the JSON mapping file at :attr:`GERRIT_CONFIGURATION.gerrit_info_path`
    and delegates to :func:`aggregate_gerrit_ids_between`.

    Args:
        start_run: Lower-bound run ID (exclusive).
        end_run: Upper-bound run ID (inclusive).
        unique: Deduplicate IDs.
        preserve_run_order: Process runs in ascending datetime order.

    Returns:
        List of aggregated Gerrit change IDs.
    """
    with open(GERRIT_CONFIGURATION.gerrit_info_path, "r") as fh:
        run_to_ids: Dict[str, List[int]] = json.loads(fh.read())

    all_ids, selected_runs, _ = aggregate_gerrit_ids_between(
        run_to_ids, start_run, end_run, unique=unique, preserve_run_order=preserve_run_order
    )
    logger.info(f"Found {len(all_ids)} Gerrit IDs between {start_run} and {end_run}")
    logger.info(f"Selected runs: {selected_runs}")
    return all_ids


# ---------------------------------------------------------------------------
# Change-detail processing
# ---------------------------------------------------------------------------


async def _process_gerrit_id(client: GerritClientAsync, change_id: int) -> Optional[Dict]:
    """Fetch and normalise a single Gerrit change detail for report generation.

    Args:
        client: An open :class:`GerritClientAsync` session.
        change_id: Numeric Gerrit change number.

    Returns:
        Dict with keys ``commit_url``, ``commit_message``, ``repository_name``,
        ``gerrit_raised_by``, ``gerrit_reviewed_by``, ``gerrit_approved_by``;
        or ``None`` if the API call fails or returns unexpected data.
    """
    response = await client.get_change_detail(change_id)
    if not isinstance(response, dict):
        return None

    owner = response.get("owner", {})
    result: Dict[str, Any] = {
        "commit_url": f"{GERRIT_API_CONFIG.host}/{change_id}",
        "commit_message": response.get("subject", ""),
        "repository_name": response.get("project", ""),
        "gerrit_raised_by": [
            {
                "name": owner.get("name", ""),
                "email": owner.get("email", ""),
                "QC_name": owner.get("username", ""),
            }
        ],
    }

    raised_by_username = owner.get("username", "")
    reviewed_by, approved_by = [], []

    all_reviewers = response.get("labels", {}).get("Code-Review", {}).get("all", []) + response.get("labels", {}).get(
        "Developer-Verified", {}
    ).get("all", [])
    for reviewer in all_reviewers:
        value = reviewer.get("value", 0)
        if not value or "username" not in reviewer:
            continue
        info = {
            "name": reviewer.get("name"),
            "email": reviewer.get("email"),
            "QC_name": reviewer.get("username"),
        }
        if value == 2:
            approved_by.append(info)
        elif value == 1 and reviewer["username"] != raised_by_username:
            reviewed_by.append(info)

    result["gerrit_reviewed_by"] = reviewed_by
    result["gerrit_approved_by"] = approved_by
    return result


async def get_gerrit_info_between_2_runids(run_id_a: str, run_id_b: str) -> List[Dict]:
    """Fetch change details for all Gerrit IDs between two run IDs.

    Args:
        run_id_a: Start run ID (exclusive lower bound).
        run_id_b: End run ID (inclusive upper bound).

    Returns:
        List of change-detail dicts (``None`` results from failed API calls
        are filtered out).
    """
    all_gerrit_ids = aggregate_gerrit_ids_for_range_string(run_id_a, run_id_b)
    async with GerritClientAsync() as client:
        results = await asyncio.gather(*[_process_gerrit_id(client, gid) for gid in all_gerrit_ids])
    return [r for r in results if r is not None]


async def get_regression_gerrits_based_of_type(
    run_id_a: str,
    run_id_b: str,
    type_backend_mapping: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """Fetch and group Gerrit changes by backend type for regression reporting.

    Each change is mapped to one or more backend types based on the repository
    name and :attr:`GERRIT_CONFIGURATION.gerrit_backend_configuration`.
    Changes whose repository does not match any configured backend are placed
    under ``"common"``.

    Args:
        run_id_a: Start run ID.
        run_id_b: End run ID.
        type_backend_mapping: Dict mapping test type → list of backend names
            (used to group changes for the report).

    Returns:
        Nested dict ``{test_type: {backend: [change_detail_dicts]}}``.
    """
    gerrit_changes = await get_gerrit_info_between_2_runids(run_id_a, run_id_b)
    by_backend: Dict[str, List[Dict]] = defaultdict(list)

    for change in gerrit_changes:
        repo = change.get("repository_name", "")
        if repo:
            repo = repo.split("/")[1].lower()

        backends = [
            bck
            for bck, configs in GERRIT_CONFIGURATION.gerrit_backend_configuration.items()
            if any(repo == cfg for cfg in configs)
        ]
        if not backends:
            backends = ["common"]

        for bck in backends:
            by_backend[bck].append(change)

    response: Dict[str, Dict] = {}
    if "common" in by_backend:
        response["common"] = {"all_runtimes": by_backend["common"]}

    for _type, backend_list in type_backend_mapping.items():
        if _type in {"converter", "quantizer"}:
            response[_type] = {_type: by_backend.get(_type, [])}
        else:
            response[_type] = {bck: by_backend.get(bck, []) for bck in backend_list}

    return response

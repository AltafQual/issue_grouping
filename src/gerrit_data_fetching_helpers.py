import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from src.async_gerrit_client import GerritClientAsync
from src.constants import GERRIT_API_CONFIG, GERRIT_CONFIGURATION
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)


def parse_candidate_datetime(token: str) -> Optional[datetime]:
    """
    Try to parse a numeric token as datetime in supported formats:
    - 14-digit YYYYmmddHHMMSS
    - 12-digit yymmddHHMMSS (by trying both ends if token is longer)
    Returns a datetime on success, else None.
    """
    # Prefer 14-digit full format if available
    if len(token) == 14:
        try:
            return datetime.strptime(token, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    # Fallback: 12-digit short format (yyMMddHHmmss)
    if len(token) >= 12:
        for cand in (token[:12], token[-12:]):
            try:
                return datetime.strptime(cand, "%y%m%d%H%M%S")
            except ValueError:
                continue

    return None


def parse_embedded_datetime(s: str) -> Optional[datetime]:
    """
    Find the first 12- to 14-digit numeric token inside 's' that parses as datetime.
    Returns a datetime or None if nothing parseable is found.
    """
    for token in re.findall(r"\d{12,14}", str(s)):
        dt = parse_candidate_datetime(token)
        if dt is not None:
            return dt
    return None


def filter_and_sort_by_embedded_datetime(strings: Iterable[str], n_remove: int = 0) -> List[str]:
    """
    Extract an embedded datetime from each string (supports 12-digit yymmddHHMMSS and 14-digit yyyymmddHHMMSS),
    sort by that datetime, remove the first `n_remove` items (earliest ones), and return the rest.
    Strings without a parseable datetime are dropped.

    Parameters:
    - strings: iterable of input strings.
    - n_remove: number of earliest items to remove after sorting.

    Returns:
    - List[str]: strings filtered and sorted by the extracted datetime (ascending).
    """
    dated = []
    for s in strings:
        dt = parse_embedded_datetime(s)
        if dt is not None:
            dated.append((dt, s))

    # Stable, deterministic: by datetime, then by string as tiebreaker
    dated.sort(key=lambda x: (x[0], x[1]))

    return [s for _, s in dated[n_remove:]]


def get_runs_between_time_limits(
    run_names: Iterable[str],
    start_run: str,
    end_run: str,
) -> List[str]:
    """
    Given a collection of run names, return those whose embedded datetime falls
    between the datetime of start_run and end_run.

    - If either start_run or end_run lacks a parseable datetime, raises ValueError.
    - Order of run_names in the result is ascending by datetime (stable with tie by name).
    - Bounds are inclusive by default; set inclusive=False for strict inequality.

    Returns:
    - List[str]: runs within [start_dt, end_dt] or (start_dt, end_dt), sorted ascending.
    """
    start_dt = parse_embedded_datetime(start_run)
    end_dt = parse_embedded_datetime(end_run)
    if start_dt is None or end_dt is None:
        raise ValueError(f"Could not parse datetimes from range endpoints: {start_run!r}, {end_run!r}")

    lo, hi = sorted((start_dt, end_dt))
    selected = []
    for r in run_names:
        dt = parse_embedded_datetime(r)
        if dt is None:
            continue
        # start date + 1 to end date
        if lo < dt <= hi:
            selected.append((dt, r))

    # Deterministic sort: by datetime, then by string
    selected.sort(key=lambda x: (x[0], x[1]))
    return [r for _, r in selected]


def aggregate_gerrit_ids_between(
    run_to_ids: Dict[str, List[int]],
    start_run: str,
    end_run: str,
    unique: bool = True,
    preserve_run_order: bool = True,
) -> Tuple[List[int], List[str], Dict[str, List[int]]]:
    """
    Aggregate Gerrit IDs for all runs whose embedded datetime lies between start_run and end_run.

    Parameters:
    - run_to_ids: dict mapping run name -> list of Gerrit IDs (may contain duplicates).
    - start_run, end_run: the range endpoints (strings containing embedded datetimes).
    - inclusive: include endpoints (default True).
    - unique: if True, deduplicate IDs while preserving first occurrence order.
    - preserve_run_order: if True, runs are processed in ascending datetime order; if False, descending.

    Returns:
    - all_ids: aggregated Gerrit IDs list (deduped if unique=True).
    - selected_runs: list of run names included (sorted ascending/descending by datetime).
    - ids_by_run: dict of {run_name: IDs used for that run} after optional dedup filtering.
    """
    # Determine selected runs between limits
    selected_runs = get_runs_between_time_limits(run_to_ids.keys(), start_run, end_run)
    if not preserve_run_order:
        selected_runs = list(reversed(selected_runs))

    # Aggregate IDs
    all_ids: List[int] = []
    seen = set()
    ids_by_run: Dict[str, List[int]] = {}

    for run in selected_runs:
        ids = run_to_ids.get(run, [])
        if unique:
            filtered = []
            for gid in ids:
                if gid not in seen:
                    seen.add(gid)
                    filtered.append(gid)
            ids_by_run[run] = filtered
            all_ids.extend(filtered)
        else:
            ids_by_run[run] = list(ids)
            all_ids.extend(ids)

    return all_ids, selected_runs, ids_by_run


def aggregate_gerrit_ids_for_range_string(
    start_run: str,
    end_run: str,
    unique: bool = True,
    preserve_run_order: bool = True,
) -> Tuple[List[int], List[str], Dict[str, List[int]]]:
    # read gerrit information
    run_to_ids = {}
    with open(GERRIT_CONFIGURATION.gerrit_info_path, "r") as f:
        run_to_ids = json.loads(f.read())

    all_ids, selected_runs, ids_by_run = aggregate_gerrit_ids_between(
        run_to_ids,
        start_run,
        end_run,
        unique=unique,
        preserve_run_order=preserve_run_order,
    )
    logger.info(f"Found gerrits: {len(all_ids)} merged between {start_run} and {end_run}")
    logger.info(f"Run selected: {selected_runs}")
    return all_ids


async def _process_gerrit_id(client: GerritClientAsync, _id: int) -> Optional[Dict]:
    """Helper to fetch and process a single Gerrit ID."""
    response = await client.get_change_detail(_id)
    if not isinstance(response, dict):
        return None

    inner_response = {
        "commit_url": f"{GERRIT_API_CONFIG.host}/{_id}",
        "commit_message": response.get("subject", ""),
        "repository_name": response.get("project", ""),
        "gerrit_raised_by": [
            {
                "name": response.get("owner", {}).get("name", ""),
                "email": response.get("owner", {}).get("email", ""),
                "QC_name": response.get("owner", {}).get("username", ""),
            }
        ],
    }

    gerrit_raised_by = response.get("owner", {}).get("username", "")
    gerrit_reviewed_by = []
    gerrit_merged_by = []

    all_reviewers = response.get("labels", {}).get("Code-Review", {}).get("all", []) + response.get("labels", {}).get(
        "Developer-Verified", {}
    ).get("all", [])

    for reviewer in all_reviewers:
        if reviewer.get("value", 0) and "username" in reviewer:
            reviewer_info = {
                "name": reviewer.get("name"),
                "email": reviewer.get("email"),
                "QC_name": reviewer.get("username"),
            }
            if reviewer["value"] == 2:
                gerrit_merged_by.append(reviewer_info)
            elif reviewer["value"] == 1 and reviewer["username"] != gerrit_raised_by:
                gerrit_reviewed_by.append(reviewer_info)

    inner_response["gerrit_reviewed_by"] = gerrit_reviewed_by
    inner_response["gerrit_merged_by"] = gerrit_merged_by
    return inner_response


async def get_gerrit_info_between_2_runids(run_id_a, run_id_b):
    all_gerrit_ids = aggregate_gerrit_ids_for_range_string(run_id_a, run_id_b)

    async with GerritClientAsync() as client:
        tasks = [_process_gerrit_id(client, gerrit_id) for gerrit_id in all_gerrit_ids]
        results = await asyncio.gather(*tasks)

    # Filter out None results from failed API calls
    return [result for result in results if result is not None]


async def get_regression_gerrits_based_of_type(run_id_a, run_id_b, type_backend_mapping):
    gerrits_data_list = await get_gerrit_info_between_2_runids(run_id_a, run_id_b)

    backend_based_gerrit_data = defaultdict(list)
    for gerrit_data in gerrits_data_list:
        repo_name = gerrit_data.get("repository_name", "")
        if repo_name:
            repo_name = repo_name.split("/")[1].lower()
        backend_names = []
        for list_of_config, bck_name in GERRIT_CONFIGURATION.gerrit_backend_configuration.items():
            if repo_name in list_of_config:
                backend_names.append(bck_name)
        for backend_name in backend_names:
            backend_based_gerrit_data[backend_name].append(gerrit_data)

    response_data = defaultdict(list)
    if "common" in backend_based_gerrit_data:
        response_data["common"] = backend_based_gerrit_data["common"]

    for _type, backend_list in type_backend_mapping.items():
        gerrit_data_dict = {}
        if _type in {"converter", "quantizer"}:
            gerrit_data_dict[_type] = backend_based_gerrit_data.get(_type, [])
        else:
            for backend in backend_list:
                gerrit_data_dict[backend] = backend_based_gerrit_data.get(backend, [])
        response_data[_type].append(gerrit_data_dict)

    return response_data

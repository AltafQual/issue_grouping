"""Previous test-plan ID lookup utilities.

Provides helpers for querying the secondary MySQL database (via SQLAlchemy)
to find the previous test-plan ID for a given run, which is used by the
regression and consolidated-report pipelines.

Layering
--------
This module sits in the **utils** layer.  It imports from ``src.constants``
and standard library / third-party packages only.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from yaml.loader import SafeLoader

from src.constants import CONSOLIDATED_REPORTS
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)
pd.set_option("future.no_silent_downcasting", True)

__all__ = [
    "read_yaml",
    "get_mysql_db_info",
    "parse_performance_score",
    "parse_testplan_id",
    "get_testplan_execution_status_df",
    "get_valid_previous_testplan_id",
    "get_valid_previous_release_testplan_id",
    "get_previous_release_testplan_id",
    "get_previous_testplan_id",
    "get_all_testplans_df",
    "get_previous_testplan_id_from_df",
    "get_previous_release_testplan_id_from_df",
    "iterate_db_get_testplan",
    "filter_and_sort_by_embedded_datetime",
    "extract_embedded_datetime",
    "filter_run_ids_within_days",
    "get_bu_name",
]


def read_yaml(yaml_config_file):
    """Read YAML file & return the data"""
    with open(yaml_config_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data


def get_mysql_db_info(mysql_info):
    """Return MySQL connection info tuple."""
    return (
        mysql_info["Username"],
        mysql_info["Password"],
        mysql_info["Port"],
        mysql_info["DatabaseName"],
        mysql_info["HostName"],
    )


def parse_performance_score(score, product="SNPE"):
    """Parse performance score data and extract inference/init/deinit times.

    Args:
        score: Performance score data (JSON string or dict).
        product: Product type — ``"SNPE"`` or ``"QNN"``.

    Returns:
        Tuple ``(inf_time, init_time, deinit_time)`` with ``None`` for missing values.
    """
    inf_time = init_time = deinit_time = None
    try:
        if score is None:
            return inf_time, init_time, deinit_time
        if isinstance(score, str):
            try:
                score_dict = json.loads(str(score).replace("'", '"').replace("None", '"None"'))
            except (json.JSONDecodeError, TypeError):
                return inf_time, init_time, deinit_time
        elif isinstance(score, dict):
            score_dict = score
        else:
            return inf_time, init_time, deinit_time

        if "Execution_Data" not in score_dict:
            return inf_time, init_time, deinit_time
        execution_data = score_dict.get("Execution_Data", {})
        if not execution_data:
            return inf_time, init_time, deinit_time
        runtime_key = next(iter(execution_data), None)
        if not runtime_key:
            return inf_time, init_time, deinit_time
        runtime_data = execution_data[runtime_key]

        if product.upper() == "SNPE":
            if (
                "Total Inference Time 75P" in runtime_data
                and runtime_data["Total Inference Time 75P"].get("Avg_Time") is not None
            ):
                inf_time = runtime_data["Total Inference Time 75P"].get("Avg_Time")
            elif (
                "Total Inference Time" in runtime_data
                and runtime_data["Total Inference Time"].get("Avg_Time") is not None
            ):
                inf_time = runtime_data["Total Inference Time"].get("Avg_Time")
            if "Init" in runtime_data and runtime_data["Init"].get("Min_Time") is not None:
                init_time = runtime_data["Init"].get("Min_Time")
            if "De-Init" in runtime_data and runtime_data["De-Init"].get("Min_Time") is not None:
                deinit_time = runtime_data["De-Init"].get("Min_Time")
        elif product.upper() == "QNN":
            if (
                "Total Inference Time 75P" in runtime_data
                and runtime_data["Total Inference Time 75P"].get("Avg_Time") is not None
            ):
                inf_time = runtime_data["Total Inference Time 75P"].get("Avg_Time")
            elif (
                "Total Inference Time [NetRun]" in runtime_data
                and runtime_data["Total Inference Time [NetRun]"].get("Avg_Time") is not None
            ):
                inf_time = runtime_data["Total Inference Time [NetRun]"].get("Avg_Time")
            if (
                "Init Stats [NetRun]" in runtime_data
                and runtime_data["Init Stats [NetRun]"].get("Min_Time") is not None
            ):
                init_time = runtime_data["Init Stats [NetRun]"].get("Min_Time")
            if (
                "De-Init Stats [NetRun]" in runtime_data
                and runtime_data["De-Init Stats [NetRun]"].get("Min_Time") is not None
            ):
                deinit_time = runtime_data["De-Init Stats [NetRun]"].get("Min_Time")
    except Exception as e:
        logger.info(e)
    return inf_time, init_time, deinit_time


def parse_testplan_id(testplan_id):
    """Parse a testplan_id string into its components dict.

    Args:
        testplan_id: A string like ``SNPE-v2.40.0.250920042144_127164-pt_alt``.

    Returns:
        Dict with keys: ``prefix``, ``version``, ``timestamp``, ``build_number``,
        ``suffix``, ``suffix_base``, ``is_rc``.

    Raises:
        ValueError: If the string doesn't match any known format.
    """
    pattern1 = r"^(SNPE|QNN|QAIRT|QNN_DELEGATE)-v([\d\.]+)\.?(\d+)_(\d+)-(.+)$"
    match = re.match(pattern1, testplan_id)
    if match:
        prefix, version, timestamp, build_number, suffix = match.groups()
        return {
            "prefix": prefix,
            "version": version,
            "timestamp": timestamp,
            "build_number": build_number,
            "suffix": suffix,
            "suffix_base": suffix.split("_RC")[0] if "_RC" in suffix else suffix,
            "is_rc": "_RC" in suffix,
        }

    pattern2 = r"^(SNPE|QNN|QAIRT|QNN_DELEGATE)-v([\d\.]+)\.?(\d+)_(\d+)_win-(.+)$"
    match = re.match(pattern2, testplan_id)
    if match:
        prefix, version, timestamp, build_number, suffix = match.groups()
        return {
            "prefix": prefix,
            "version": version,
            "timestamp": timestamp,
            "build_number": f"{build_number}_win",
            "suffix": suffix,
            "suffix_base": suffix.split("_RC")[0] if "_RC" in suffix else suffix,
            "is_rc": "_RC" in suffix,
        }

    pattern3 = r"^(SNPE|QNN|QAIRT|QNN_DELEGATE)-v([\d\.]+)\.?(\d+)-(.+)$"
    match = re.match(pattern3, testplan_id)
    if match:
        prefix, version, timestamp, suffix = match.groups()
        return {
            "prefix": prefix,
            "version": version,
            "timestamp": timestamp,
            "build_number": "",
            "suffix": suffix,
            "suffix_base": suffix.split("_RC")[0] if "_RC" in suffix else suffix,
            "is_rc": "_RC" in suffix,
        }

    if "-v" in testplan_id:
        parts = testplan_id.split("-v", 1)
        prefix = parts[0]
        rest = parts[1]
        version_parts = re.match(r"([\d\.]+)\.?(\d+)_", rest)
        if version_parts:
            version = version_parts.group(1)
            timestamp = version_parts.group(2)
            rest = rest[len(version) + 1 + len(timestamp) + 1 :]
            if "-" in rest:
                build_parts = rest.split("-", 1)
                build_number = build_parts[0]
                suffix = build_parts[1]
            else:
                build_number = rest.split("_")[0]
                suffix = "_".join(rest.split("_")[1:])
            return {
                "prefix": prefix,
                "version": version,
                "timestamp": timestamp,
                "build_number": build_number,
                "suffix": suffix,
                "suffix_base": suffix.split("_RC")[0] if "_RC" in suffix else suffix,
                "is_rc": "_RC" in suffix,
            }

    raise ValueError(f"Invalid testplan_id format: {testplan_id}")


def get_testplan_execution_status_df(session, testplan_id, debug=False):
    """Return ``(completion_pct, df)`` for *testplan_id* from the DB.

    Args:
        session: SQLAlchemy session.
        testplan_id: Test plan ID to check.
        debug: Log extra info when ``True``.

    Returns:
        ``(int percentage, DataFrame)`` or ``(None, None)`` on error.
    """
    try:
        query = text(
            "SELECT tc_uuid, exec_status, result, score, product_name FROM result WHERE testplan_id = :testplan_id"
        )
        results = session.execute(query, {"testplan_id": testplan_id}).fetchall()
        if not results:
            return None, None

        df = pd.DataFrame(results, columns=["tc_uuid", "exec_status", "result", "score", "product_name"])
        product = "SNPE" if "snpe" in testplan_id.lower() else "QNN"
        try:
            df[["inf", "init", "de_init"]] = df.apply(
                lambda row: parse_performance_score(row["score"], product), axis=1, result_type="expand"
            )
        except Exception as e:
            logger.info(f"Error parsing performance scores: {e}")
            df["inf"] = df["init"] = df["de_init"] = None
        df.fillna(np.nan, inplace=True)

        total_count = len(df)
        completed_count = len(df[df["exec_status"] == "COMPLETED"])
        if total_count > 0:
            return int((completed_count / total_count) * 100), df
        return 0, df
    except Exception as e:
        logger.info(f"Error getting execution status: {e}")
        return None, None


def get_previous_testplan_id(session, testplan_id, debug=False):
    """Find the immediately preceding testplan ID for *testplan_id*."""
    try:
        components = parse_testplan_id(testplan_id)
        suffix_pattern = f"%-{components['suffix_base']}_RC%" if components["is_rc"] else f"%-{components['suffix']}"
        query = text(
            "SELECT testplan_id FROM result WHERE testplan_id LIKE :prefix_pattern "
            "AND testplan_id LIKE :suffix_pattern AND testplan_id < :current_testplan_id "
            "GROUP BY testplan_id ORDER BY testplan_id DESC LIMIT 1"
        )
        for prefix_pattern in [
            f"{components['prefix']}-v{components['version']}%",
            f"{components['prefix']}-%",
        ]:
            result = session.execute(
                query,
                {
                    "prefix_pattern": prefix_pattern,
                    "suffix_pattern": suffix_pattern,
                    "current_testplan_id": testplan_id,
                },
            ).fetchone()
            if result:
                return result[0]
        return None
    except Exception as e:
        logger.info(f"Error querying database: {e}")
        return None


def get_valid_previous_testplan_id(session, testplan_id, min_exec_status=90, debug=False):
    """Find the most recent preceding testplan ID with execution status >= *min_exec_status*."""
    try:
        components = parse_testplan_id(testplan_id)
        suffix_pattern = f"%-{components['suffix_base']}_RC%" if components["is_rc"] else f"%-{components['suffix']}"
        query = text(
            "SELECT testplan_id FROM result WHERE testplan_id LIKE :prefix_pattern "
            "AND testplan_id LIKE :suffix_pattern AND testplan_id < :current_testplan_id "
            "GROUP BY testplan_id ORDER BY testplan_id DESC"
        )
        results = session.execute(
            query,
            {
                "prefix_pattern": f"{components['prefix']}-%",
                "suffix_pattern": suffix_pattern,
                "current_testplan_id": testplan_id,
            },
        ).fetchall()
        for row in results:
            exec_status, _ = get_testplan_execution_status_df(session, row[0], debug)
            if exec_status is not None and exec_status >= min_exec_status:
                return row[0]
        return None
    except Exception as e:
        logger.info(f"Error finding valid previous testplan_id: {e}")
        return None


def get_previous_release_testplan_id(session, testplan_id, debug=False):
    """Find the previous release (RC) testplan ID for *testplan_id*."""
    try:
        components = parse_testplan_id(testplan_id)
        version_parts = components["version"].split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        base_suffix = components["suffix"].split("_RC")[0] if "_RC" in components["suffix"] else components["suffix"]

        query = text(
            "SELECT testplan_id FROM result WHERE testplan_id LIKE :prefix_pattern "
            "AND testplan_id LIKE :suffix_pattern GROUP BY testplan_id ORDER BY testplan_id DESC"
        )
        results = session.execute(
            query,
            {"prefix_pattern": f"{components['prefix']}-%", "suffix_pattern": f"%-{base_suffix}_RC%"},
        ).fetchall()

        version_rc_testplans: dict = {}
        for row in results:
            testplan = row[0]
            vm = re.search(r"-v([\d\.]+)", testplan)
            if not vm:
                continue
            vp = vm.group(1).split(".")
            if len(vp) < 2:
                continue
            vm_major, vm_minor = int(vp[0]), int(vp[1])
            if vm_major < major or (vm_major == major and vm_minor < minor):
                rc_m = re.search(r"_RC(\d+)$", testplan)
                if rc_m:
                    rc_num = int(rc_m.group(1))
                    ver_key = f"{vm_major}.{vm_minor}"
                    if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                        version_rc_testplans[ver_key] = (rc_num, testplan)

        if not version_rc_testplans:
            return None
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )
        return version_rc_testplans[sorted_versions[0]][1]
    except Exception as e:
        logger.info(f"Error finding previous release testplan_id: {e}")
        return None


def get_valid_previous_release_testplan_id(session, testplan_id, min_exec_status=90, debug=False):
    """Find the previous release testplan ID with execution status >= *min_exec_status*."""
    try:
        components = parse_testplan_id(testplan_id)
        version_parts = components["version"].split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        base_suffix = components["suffix"].split("_RC")[0] if "_RC" in components["suffix"] else components["suffix"]

        query = text(
            "SELECT testplan_id FROM result WHERE testplan_id LIKE :prefix_pattern "
            "AND testplan_id LIKE :suffix_pattern GROUP BY testplan_id ORDER BY testplan_id DESC"
        )
        results = session.execute(
            query,
            {"prefix_pattern": f"{components['prefix']}-%", "suffix_pattern": f"%-{base_suffix}_RC%"},
        ).fetchall()

        version_rc_testplans: dict = {}
        for row in results:
            testplan = row[0]
            vm = re.search(r"-v([\d\.]+)", testplan)
            if not vm:
                continue
            vp = vm.group(1).split(".")
            if len(vp) < 2:
                continue
            vm_major, vm_minor = int(vp[0]), int(vp[1])
            if (vm_major < major or (vm_major == major and vm_minor < minor)) and len(vp) == 2:
                exec_status, _ = get_testplan_execution_status_df(session, testplan, debug)
                if exec_status is None or exec_status < min_exec_status:
                    continue
                rc_m = re.search(r"_RC(\d+)$", testplan)
                if rc_m:
                    rc_num = int(rc_m.group(1))
                    ver_key = f"{vm_major}.{vm_minor}"
                    if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                        version_rc_testplans[ver_key] = (rc_num, testplan, exec_status)

        if not version_rc_testplans:
            return None
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )
        return version_rc_testplans[sorted_versions[0]][1]
    except Exception as e:
        logger.info(f"Error finding valid previous release testplan_id: {e}")
        return None


def get_all_testplans_df(session, debug=False):
    """Return a DataFrame of all distinct testplan IDs from the DB."""
    try:
        results = session.execute(text("SELECT DISTINCT testplan_id FROM result ORDER BY testplan_id DESC")).fetchall()
        return pd.DataFrame(results, columns=["testplan_id"])
    except Exception as e:
        logger.info(f"Error getting all testplan_ids: {e}")
        return pd.DataFrame(columns=["testplan_id"])


def get_previous_testplan_id_from_df(df, testplan_id, debug=False):
    """Find the previous testplan ID from an in-memory DataFrame of all IDs."""
    try:
        components = parse_testplan_id(testplan_id)
        prefix = components["prefix"]
        version = components["version"]

        def _make_mask(prefix_filter, suffix_filter):
            return prefix_filter & suffix_filter & (df["testplan_id"] < testplan_id)

        for prefix_filter in [
            df["testplan_id"].str.startswith(f"{prefix}-v{version}"),
            df["testplan_id"].str.startswith(f"{prefix}-"),
        ]:
            if components["is_rc"]:
                suffix_filter = df["testplan_id"].str.contains(f"-{components['suffix_base']}_RC")
            elif "_win" in components["build_number"]:
                suffix_filter = df["testplan_id"].str.contains(f"_win-{components['suffix']}")
            else:
                suffix_filter = df["testplan_id"].str.endswith(f"-{components['suffix']}")
            filtered = df[_make_mask(prefix_filter, suffix_filter)]
            if not filtered.empty:
                return filtered.iloc[0]["testplan_id"]
        return None
    except Exception as e:
        logger.info(f"Error finding previous testplan_id from DataFrame: {e}")
        return None


def get_previous_release_testplan_id_from_df(df, testplan_id, debug=False):
    """Find the previous release testplan ID from an in-memory DataFrame of all IDs."""
    try:
        components = parse_testplan_id(testplan_id)
        version_parts = components["version"].split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1])
        base_suffix = components["suffix"].split("_RC")[0] if "_RC" in components["suffix"] else components["suffix"]
        prefix = components["prefix"]

        mask = df["testplan_id"].str.startswith(f"{prefix}-") & df["testplan_id"].str.contains(f"-{base_suffix}_RC")
        filtered_df = df[mask]
        if filtered_df.empty:
            return None

        version_rc_testplans: dict = {}
        for _, row in filtered_df.iterrows():
            testplan = row["testplan_id"]
            vm = re.search(r"-v([\d\.]+)", testplan)
            if not vm:
                continue
            vp = vm.group(1).split(".")
            if len(vp) < 2:
                continue
            vm_major, vm_minor = int(vp[0]), int(vp[1])
            if vm_major < major or (vm_major == major and vm_minor < minor):
                rc_m = re.search(r"_RC(\d+)$", testplan)
                if rc_m:
                    rc_num = int(rc_m.group(1))
                    ver_key = f"{vm_major}.{vm_minor}"
                    if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                        version_rc_testplans[ver_key] = (rc_num, testplan)

        if not version_rc_testplans:
            return None
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )
        return version_rc_testplans[sorted_versions[0]][1]
    except Exception as e:
        logger.info(f"Error finding previous release testplan_id from DataFrame: {e}")
        return None


def iterate_db_get_testplan(testplan_id):
    """Connect to the secondary MySQL DB and return previous testplan dataframes.

    Args:
        testplan_id: Current testplan ID string.

    Returns:
        Tuple ``(p_n_df, p_r_df, previous_testplan_id, previous_release_testplan_id)``.
    """
    yaml_file = CONSOLIDATED_REPORTS.qa2_config_file_path
    p_n_df, p_r_df, previous_testplan_id, previous_release_testplan_id = (pd.DataFrame(), pd.DataFrame(), None, None)

    if not yaml_file:
        logger.info("Error: QA2_CONFIG_FILE_PATH environment variable not set.")
        sys.exit(1)

    try:
        yaml_info = read_yaml(yaml_file)
        mysql_info = yaml_info["Database"]
        username, password, port, database, hostname = get_mysql_db_info(mysql_info)
        uri = f"mysql+pymysql://{username}:{quote_plus(password)}@{hostname}:{port}/{database}"
        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        df = get_all_testplans_df(session)
        previous_testplan_id = get_previous_testplan_id_from_df(df, testplan_id)
        previous_release_testplan_id = get_previous_release_testplan_id_from_df(df, testplan_id)

        if previous_testplan_id:
            exec_status, p_n_df = get_testplan_execution_status_df(session, previous_testplan_id)
            if exec_status is None or exec_status < 90:
                valid = get_valid_previous_testplan_id(session, testplan_id, 90)
                if valid:
                    _, p_n_df = get_testplan_execution_status_df(session, valid)

        if previous_release_testplan_id:
            exec_status, p_r_df = get_testplan_execution_status_df(session, previous_release_testplan_id)
            if exec_status is None or exec_status < 90:
                valid = get_valid_previous_release_testplan_id(session, testplan_id, 90)
                if valid:
                    _, p_r_df = get_testplan_execution_status_df(session, valid)

    except Exception as e:
        logger.info(f"Error: {e}")
        sys.exit(1)
    finally:
        if "session" in locals():
            session.close()

    return p_n_df, p_r_df, previous_testplan_id, previous_release_testplan_id


# ---------------------------------------------------------------------------
# Extra utilities added in the new arch
# ---------------------------------------------------------------------------


def filter_and_sort_by_embedded_datetime(strings: list[str], n_remove: int = 0) -> list[str]:
    """Sort strings by an embedded datetime and drop the first *n_remove* entries.

    Supports 12-digit ``yymmddHHMMSS`` and 14-digit ``yyyymmddHHMMSS`` tokens.
    Strings without a parseable datetime are dropped.

    Args:
        strings: List of input strings.
        n_remove: Number of earliest items to remove after sorting.

    Returns:
        List of strings sorted by embedded datetime with earliest *n_remove* removed.
    """
    dated = [(extract_embedded_datetime(s), s) for s in strings]
    dated = [t for t in dated if t[0] is not None]
    dated.sort(key=lambda x: (x[0], x[1]))
    return [s for _, s in dated[n_remove:]]


def extract_embedded_datetime(run_id: str) -> datetime | None:
    """Return the datetime embedded in *run_id*, or ``None`` if no token parses.

    Supports 12-digit ``yymmddHHMMSS`` and 14-digit ``yyyymmddHHMMSS`` tokens.
    """

    def parse_candidate(token: str):
        if len(token) == 14:
            try:
                return datetime.strptime(token, "%Y%m%d%H%M%S")
            except ValueError:
                pass
        if len(token) >= 12:
            for cand in (token[:12], token[-12:]):
                try:
                    return datetime.strptime(cand, "%y%m%d%H%M%S")
                except ValueError:
                    continue
        return None

    for token in re.findall(r"\d{12,14}", str(run_id)):
        dt = parse_candidate(token)
        if dt is not None:
            return dt
    return None


def filter_run_ids_within_days(run_ids: list[str], days: int) -> list[str]:
    """Return run_ids whose embedded datetime is within the last *days* days.

    Run IDs that lack a parseable datetime are dropped (same policy as
    :func:`filter_and_sort_by_embedded_datetime`).
    """
    cutoff = datetime.now() - timedelta(days=days)
    return [rid for rid in run_ids if (dt := extract_embedded_datetime(rid)) is not None and dt >= cutoff]


def get_bu_name(soc_name: str) -> str:
    """Map a SoC name to a business unit abbreviation.

    Args:
        soc_name: SoC identifier string (e.g. ``"Kodiak"``, ``"QCS8550"``).

    Returns:
        BU label: one of ``"AUTO"``, ``"CBN"``, ``"Compute"``, ``"Tools"``,
        ``"IOT"``, ``"Mobile"``, ``"Wearables"``, ``"XR"``, or ``"Unknown"``.
    """
    auto = {
        "Lemans_QOS224Q",
        "LemansIVI",
        "LemansQ",
        "Makena8295Q",
        "MakenaIVI",
        "MakenaQ",
        "Monaco7775Q",
        "Monaco8620LE",
        "Monaco_qnx710Q",
        "MonacoQ",
        "NordLE",
        "QCM8538",
        "AIC100_x86",
    }
    cbn = {"KobukLE"}
    compute = {"GlymurW", "HamoaW", "KodiakW", "MakenaW", "PoipuW", "PurwaW", "windowshost"}
    host = {"host"}
    iot = {
        "ClarenceIOT",
        "DivarIOT",
        "KamortaIOT",
        "KodiakIOT",
        "KodiakIOTU",
        "KodiakIOTU2",
        "KodiakWIOT",
        "LemansLE",
        "MilosIOT",
        "QCM2290",
        "QCM4290A1",
        "QCM4290A3",
        "QCM6125",
        "QCM6490K2L",
        "QCM6490LE",
        "QCS410",
        "QCS410LE2",
        "QCS610",
        "QCS610LE",
        "QCS610LE2",
        "QCS610LE3",
        "QCS615LE",
        "QCS7230LE",
        "QCS8250",
        "QCS8300K2L",
        "QCS8300LE",
        "QCS8550",
        "QCS8550LE",
        "QCS8550N",
        "QCS8550U",
        "QCS8625",
        "QCS8625LE",
        "QCS9100K2L",
        "QCS9100LE",
        "QCS9100LEC1",
        "QRB4210LE",
        "QRB5165LE2",
        "QRB5165U2",
    }
    mobile = {
        "Bitra",
        "Bonito",
        "Camano",
        "Clarence",
        "Divar",
        "Eliza",
        "Fillmore",
        "Fraser",
        "Kaanapali",
        "Kailua",
        "Kalpeni",
        "Kam",
        "Kamorta",
        "Kodiak",
        "Kona",
        "Lahaina",
        "LahainaPro",
        "Lamma",
        "Lanai",
        "Mannar",
        "Milos",
        "Molokai",
        "Netrani",
        "Pakala",
        "Palawan",
        "Palima",
        "Strait",
        "Tofino",
        "Waipio",
    }
    wearables = {"AspenLAW", "AspenLW"}
    xr = {"Aurora", "AuroraLE", "AuroraLE2", "Balsam", "Halliday", "Halliday2", "Luna", "Matrix", "WaipioLE"}

    if soc_name in auto:
        return "AUTO"
    elif soc_name in cbn:
        return "CBN"
    elif soc_name in compute:
        return "Compute"
    elif soc_name in host:
        return "Tools"
    elif soc_name in iot:
        return "IOT"
    elif soc_name in mobile:
        return "Mobile"
    elif soc_name in wearables:
        return "Wearables"
    elif soc_name in xr:
        return "XR"
    else:
        return "Unknown"

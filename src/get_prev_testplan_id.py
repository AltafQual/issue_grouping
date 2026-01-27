import re
import sys
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from yaml.loader import SafeLoader

from src.constants import CONSOLIDATED_REPORTS

pd.set_option("future.no_silent_downcasting", True)


def read_yaml(yaml_config_file):
    """Read YAML file & return the data"""
    with open(yaml_config_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data


def get_mysql_db_info(mysql_info):
    """Return MySQL information"""
    return (
        mysql_info["Username"],
        mysql_info["Password"],
        mysql_info["Port"],
        mysql_info["DatabaseName"],
        mysql_info["HostName"],
    )


def parse_performance_score(score, product="SNPE"):
    """
    Parse performance score data and extract specific metrics based on product type.

    Args:
        score: The performance score data (JSON string or dictionary)
        product: The product type, either "SNPE" or "QNN" (default: "SNPE")

    Returns:
        tuple: A tuple containing (inference_time, init_time, deinit_time), with None for missing values
    """
    import json

    # Default return values
    inf_time = None
    init_time = None
    deinit_time = None

    try:
        # Check if score is None
        if score is None:
            return inf_time, init_time, deinit_time
        # print("herrrr3")
        # Convert to dictionary if it's a JSON string
        if isinstance(score, str):
            try:
                score_dict = json.loads(str(score).replace("'", '"').replace("None", '"None"'))
            except (json.JSONDecodeError, TypeError):
                return inf_time, init_time, deinit_time
        elif isinstance(score, dict):
            score_dict = score
        else:
            return inf_time, init_time, deinit_time
        # Check if Execution_Data exists
        if "Execution_Data" not in score_dict:
            return inf_time, init_time, deinit_time

        execution_data = score_dict.get("Execution_Data", {})
        if not execution_data:
            return inf_time, init_time, deinit_time
        # Find the first runtime key (e.g., "HTP_FP16")
        runtime_key = next(iter(execution_data), None)
        if not runtime_key:
            return inf_time, init_time, deinit_time

        runtime_data = execution_data[runtime_key]
        # Extract metrics based on product type
        if product.upper() == "SNPE":
            # Extract Total Inference Time 75P if available, otherwise Total Inference Time
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

            # Extract Init time
            if "Init" in runtime_data and runtime_data["Init"].get("Min_Time") is not None:
                init_time = runtime_data["Init"].get("Min_Time")

            # Extract De-Init time
            if "De-Init" in runtime_data and runtime_data["De-Init"].get("Min_Time") is not None:
                deinit_time = runtime_data["De-Init"].get("Min_Time")

        elif product.upper() == "QNN":
            # Extract Total Inference Time 75P if available, otherwise Total Inference Time [NetRun]
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

            # Extract Init Stats [NetRun]
            if (
                "Init Stats [NetRun]" in runtime_data
                and runtime_data["Init Stats [NetRun]"].get("Min_Time") is not None
            ):
                init_time = runtime_data["Init Stats [NetRun]"].get("Min_Time")

            # Extract De-Init Stats [NetRun]
            if (
                "De-Init Stats [NetRun]" in runtime_data
                and runtime_data["De-Init Stats [NetRun]"].get("Min_Time") is not None
            ):
                deinit_time = runtime_data["De-Init Stats [NetRun]"].get("Min_Time")

    except Exception as e:
        print(e)
        # If any exception occurs, return default values
        pass

    return inf_time, init_time, deinit_time


def parse_testplan_id(testplan_id):
    """
    Parse a testplan_id into its components.

    Examples:
    - SNPE-v2.40.0.250920042144_127164-pt_alt
    - QNN-v2.40.0.250925042002_127368_win-win_non_pt_nightly

    Returns a dictionary with the components:
    {
        'prefix': 'SNPE',
        'version': '2.40.0',
        'timestamp': '250920042144',
        'build_number': '127164',
        'suffix': 'pt_alt',
        'suffix_base': 'pt_alt'  # Same as suffix but without RC number
    }
    """
    # Try different patterns to match testplan_id components

    # Pattern 1: Standard format with hyphen separator
    # Example: SNPE-v2.40.0.250920042144_127164-pt_alt
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

    # Pattern 2: Format with _win- separator
    # Example: QNN-v2.40.0.250925042002_127368_win-win_non_pt_nightly
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

    # If no pattern matches, try a more general approach
    if "-v" in testplan_id:
        parts = testplan_id.split("-v", 1)
        prefix = parts[0]
        rest = parts[1]

        # Try to extract version and timestamp
        version_parts = re.match(r"([\d\.]+)\.?(\d+)_", rest)
        if version_parts:
            version = version_parts.group(1)
            timestamp = version_parts.group(2)

            # Extract the rest after timestamp
            rest = rest[len(version) + 1 + len(timestamp) + 1 :]

            # Try to extract build number and suffix
            if "-" in rest:
                build_parts = rest.split("-", 1)
                build_number = build_parts[0]
                suffix = build_parts[1]
            else:
                # If no clear separator, use a reasonable default
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

    # If all patterns fail, raise an error
    raise ValueError(f"Invalid testplan_id format: {testplan_id}")


def get_testplan_execution_status_df(session, testplan_id, debug=False):
    """
    Query the database to get the execution status of a testplan_id using a DataFrame.

    Args:
        session: SQLAlchemy session
        testplan_id: The testplan_id to check
        debug: Whether to print debug information

    Returns:
        The execution status as a percentage (0-100), or None if not found
    """
    try:
        if debug:
            print(f"Getting execution status for {testplan_id} using DataFrame approach")

        # Query to get all test cases and their execution status for the testplan_id
        query = text(
            """
            SELECT tc_uuid, exec_status, result, score, product_name FROM result 
            WHERE testplan_id = :testplan_id
        """
        )

        # Execute the query and fetch all results
        results = session.execute(query, {"testplan_id": testplan_id}).fetchall()

        if not results:
            if debug:
                print(f"No test cases found for {testplan_id}")
            return None, None

        # Create a DataFrame from the results
        df = pd.DataFrame(results, columns=["tc_uuid", "exec_status", "result", "score", "product_name"])

        # Determine product type
        if "snpe" in testplan_id.lower():
            product = "SNPE"
        else:
            product = "QNN"

        # Apply performance score parsing with proper error handling
        # print(product)
        try:
            # Use result_type='expand' to properly expand the tuple into columns
            df[["inf", "init", "de_init"]] = df.apply(
                lambda row: parse_performance_score(row["score"], product), axis=1, result_type="expand"
            )
        except Exception as e:
            # if debug:
            print(f"Error parsing performance scores: {e}")
            # Create empty columns if parsing fails
            df["inf"] = None
            df["init"] = None
            df["de_init"] = None
        df.fillna(np.nan, inplace=True)

        if debug:
            print(f"Found {len(df)} test cases for {testplan_id}")
            print(f"Execution status counts:\n{df['exec_status'].value_counts()}")

        # Calculate the percentage of completed test cases
        total_count = len(df)
        completed_count = len(df[df["exec_status"] == "COMPLETED"])

        if total_count > 0:
            percentage = (completed_count / total_count) * 100
            if debug:
                print(f"Execution status: {percentage:.2f}% ({completed_count}/{total_count} completed)")
            return int(percentage), df
        else:
            if debug:
                print(f"No test cases found for {testplan_id}")
            return 0, df

    except Exception as e:
        print(f"Error getting execution status: {e}", file=sys.stderr)
        return None, None


def get_valid_previous_testplan_id(session, testplan_id, min_exec_status=90, debug=False):
    """
    Find a valid previous testplan_id with execution status >= min_exec_status.

    Args:
        session: SQLAlchemy session
        testplan_id: The current testplan_id
        min_exec_status: Minimum execution status required (default: 90%)
        debug: Whether to print debug information

    Returns:
        A valid previous testplan_id or None if not found
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        # Determine the suffix pattern based on whether this is an RC testplan_id
        if components["is_rc"]:
            # For RC testplan_ids, match the base suffix and any RC number
            suffix_pattern = f"%-{components['suffix_base']}_RC%"
        else:
            # For non-RC testplan_ids, match the exact suffix
            suffix_pattern = f"%-{components['suffix']}"

        if debug:
            print(f"Looking for valid previous testplan_id with execution status >= {min_exec_status}%")
            print(f"Suffix pattern: {suffix_pattern}")

        # Query to find all previous testplan_ids with the same prefix and suffix
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :suffix_pattern 
            AND testplan_id < :current_testplan_id 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC
        """
        )

        prefix_pattern = f"{components['prefix']}-%"

        results = session.execute(
            query,
            {"prefix_pattern": prefix_pattern, "suffix_pattern": suffix_pattern, "current_testplan_id": testplan_id},
        ).fetchall()

        if not results:
            if debug:
                print(f"No previous testplan_ids found")
            return None

        # Find the first testplan_id with execution status >= min_exec_status
        for row in results:
            prev_testplan_id = row[0]

            # Get the execution status using the DataFrame approach
            exec_status, p_df = get_testplan_execution_status_df(session, prev_testplan_id, debug)

            if exec_status is not None and exec_status >= min_exec_status:
                if debug:
                    print(f"Found valid testplan_id: {prev_testplan_id} with execution status {exec_status}%")
                return prev_testplan_id
            else:
                if debug:
                    print(f"Skipping {prev_testplan_id} - Execution status: {exec_status}% (below threshold)")

        if debug:
            print(f"No testplan_id found with execution status >= {min_exec_status}%")
        return None

    except Exception as e:
        print(f"Error finding valid previous testplan_id: {e}", file=sys.stderr)
        return None


def get_valid_previous_release_testplan_id(session, testplan_id, min_exec_status=90, debug=False):
    """
    Find a valid previous release testplan_id with execution status >= min_exec_status.

    Args:
        session: SQLAlchemy session
        testplan_id: The current testplan_id
        min_exec_status: Minimum execution status required (default: 90%)
        debug: Whether to print debug information

    Returns:
        A valid previous release testplan_id or None if not found
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        # Extract the major version (e.g., 2.40 from 2.40.0)
        version_parts = components["version"].split(".")
        major_version = ".".join(version_parts[:2])  # e.g., "2.40"

        # Get the base suffix without any RC part
        base_suffix = components["suffix"]
        if "_RC" in base_suffix:
            base_suffix = base_suffix.split("_RC")[0]

        # Find all previous versions with RC testplans
        major_version_parts = major_version.split(".")
        major = int(major_version_parts[0])
        minor = int(major_version_parts[1])

        if debug:
            print(f"Looking for valid previous release testplan_id with execution status >= {min_exec_status}%")

        # Query to find all RC testplans for any previous version
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :suffix_pattern 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC
        """
        )

        prefix_pattern = f"{components['prefix']}-%"
        suffix_pattern = f"%-{base_suffix}_RC%"

        results = session.execute(
            query, {"prefix_pattern": prefix_pattern, "suffix_pattern": suffix_pattern}
        ).fetchall()

        if not results:
            if debug:
                print(f"No RC testplans found")
            return None

        # Group testplans by version and find the highest RC for each version with sufficient execution status
        version_rc_testplans = {}

        for row in results:
            testplan = row[0]

            try:
                # Extract version
                version_match = re.search(r"-v([\d\.]+)", testplan)
                if version_match:
                    version = version_match.group(1)
                    version_parts = version.split(".")
                    if len(version_parts) >= 2:
                        ver_major = int(version_parts[0])
                        ver_minor = int(version_parts[1])

                        # Only consider versions less than the current version
                        # Skip minor versions (e.g., 2.37.1) and only consider major versions (e.g., 2.38, 2.39)
                        if (ver_major < major or (ver_major == major and ver_minor < minor)) and len(
                            version_parts
                        ) == 2:
                            # Get the execution status using the DataFrame approach
                            exec_status, p_df = get_testplan_execution_status_df(session, testplan, debug)

                            # Skip testplans with insufficient execution status
                            if exec_status is None or exec_status < min_exec_status:
                                if debug:
                                    print(f"Skipping {testplan} - Execution status: {exec_status}% (below threshold)")
                                continue

                            # Extract RC number
                            rc_match = re.search(r"_RC(\d+)$", testplan)
                            if rc_match:
                                rc_num = int(rc_match.group(1))
                                ver_key = f"{ver_major}.{ver_minor}"

                                if debug:
                                    print(
                                        f"Found {testplan} - Version {ver_key} RC{rc_num} - Execution status: {exec_status}%"
                                    )

                                # Store the highest RC for each version with sufficient execution status
                                if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                                    version_rc_testplans[ver_key] = (rc_num, testplan, exec_status)
            except Exception as e:
                if debug:
                    print(f"Error parsing testplan {testplan}: {e}")
                continue

        if not version_rc_testplans:
            if debug:
                print(f"No valid RC testplans found with execution status >= {min_exec_status}%")
            return None

        # Sort versions in descending order
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )

        if debug:
            print(f"Found valid RC testplans for versions: {sorted_versions}")
            for ver in sorted_versions:
                rc_num, testplan, exec_status = version_rc_testplans[ver]
                print(f"  Version {ver}: RC{rc_num} - {testplan} - Execution status: {exec_status}%")

        # Return the highest RC testplan from the most recent version with sufficient execution status
        if sorted_versions:
            latest_version = sorted_versions[0]
            rc_num, testplan, exec_status = version_rc_testplans[latest_version]
            if debug:
                print(f"Selected version {latest_version} RC{rc_num}: {testplan} with execution status {exec_status}%")
            return testplan

        return None

    except Exception as e:
        print(f"Error finding valid previous release testplan_id: {e}", file=sys.stderr)
        return None


def get_previous_release_testplan_id(session, testplan_id, debug=False):
    """
    Query the database to find the previous release testplan_id for the given testplan_id.

    Args:
        session: SQLAlchemy session
        testplan_id: The current testplan_id
        debug: Whether to print debug information

    Returns:
        The previous release testplan_id or None if not found

    Strategy:
    1. Extract the version from the current testplan (e.g., 2.40)
    2. Find all previous versions with RC suffix (e.g., 2.39, 2.38, etc.)
    3. For each version, look for testplans with that version and with an "RC" suffix
    4. If there are multiple RC versions, pick the one with the highest RC number (e.g., RC6)
    5. Return the most recent release testplan
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        # Extract the major version (e.g., 2.40 from 2.40.0)
        version_parts = components["version"].split(".")
        major_version = ".".join(version_parts[:2])  # e.g., "2.40"

        # Get the base suffix without any RC part
        base_suffix = components["suffix"]
        if "_RC" in base_suffix:
            base_suffix = base_suffix.split("_RC")[0]

        # Find all previous versions with RC testplans
        major_version_parts = major_version.split(".")
        major = int(major_version_parts[0])
        minor = int(major_version_parts[1])

        # Query to find all RC testplans for any previous version
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :suffix_pattern 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC
        """
        )

        prefix_pattern = f"{components['prefix']}-%"
        suffix_pattern = f"%-{base_suffix}_RC%"

        if debug:
            print(f"Looking for release testplan_ids with:")
            print(f"  prefix_pattern: {prefix_pattern}")
            print(f"  suffix_pattern: {suffix_pattern}")

        results = session.execute(
            query, {"prefix_pattern": prefix_pattern, "suffix_pattern": suffix_pattern}
        ).fetchall()

        if not results:
            if debug:
                print(f"No RC testplans found")
            return None

        # Group testplans by version and find the highest RC for each version
        version_rc_testplans = {}

        for row in results:
            testplan = row[0]
            try:
                # Extract version
                version_match = re.search(r"-v([\d\.]+)", testplan)
                if version_match:
                    version = version_match.group(1)
                    version_parts = version.split(".")
                    if len(version_parts) >= 2:
                        ver_major = int(version_parts[0])
                        ver_minor = int(version_parts[1])

                        # Only consider versions less than the current version
                        if ver_major < major or (ver_major == major and ver_minor < minor):
                            # Extract RC number
                            rc_match = re.search(r"_RC(\d+)$", testplan)
                            if rc_match:
                                rc_num = int(rc_match.group(1))
                                ver_key = f"{ver_major}.{ver_minor}"

                                # Store the highest RC for each version
                                if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                                    version_rc_testplans[ver_key] = (rc_num, testplan)
            except Exception as e:
                if debug:
                    print(f"Error parsing testplan {testplan}: {e}")
                continue

        if not version_rc_testplans:
            if debug:
                print(f"No valid RC testplans found for previous versions")
            return None

        # Sort versions in descending order
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )

        if debug:
            print(f"Found RC testplans for versions: {sorted_versions}")
            for ver in sorted_versions:
                rc_num, testplan = version_rc_testplans[ver]
                print(f"  Version {ver}: RC{rc_num} - {testplan}")

        # Return the highest RC testplan from the most recent version
        if sorted_versions:
            latest_version = sorted_versions[0]
            rc_num, testplan = version_rc_testplans[latest_version]
            if debug:
                print(f"Selected version {latest_version} RC{rc_num}: {testplan}")
            return testplan

        return None

    except Exception as e:
        print(f"Error finding previous release testplan_id: {e}", file=sys.stderr)
        return None


def get_previous_testplan_id(session, testplan_id, debug=False):
    """
    Query the database to find the previous testplan_id for the given testplan_id.

    Args:
        session: SQLAlchemy session
        testplan_id: The current testplan_id
        debug: Whether to print debug information

    Returns:
        The previous testplan_id or None if not found

    Strategy:
    1. First try to find a testplan_id with the same prefix, version, and suffix
       - For RC testplan_ids, match the base suffix and any RC number
    2. If that fails, try to find a testplan_id with the same prefix and suffix
       - For RC testplan_ids, match the base suffix and any RC number
    3. If that fails, try to find a testplan_id with the same prefix, any version, and suffix
       - For RC testplan_ids, match the base suffix and any RC number
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        # Construct a query to find the previous testplan_id
        # We'll look for testplan_ids with the same prefix and suffix, but with a lower build number
        # or an earlier timestamp if the build number is the same

        # Determine the suffix pattern based on whether this is an RC testplan_id
        if components["is_rc"]:
            # For RC testplan_ids, match the base suffix and any RC number
            suffix_pattern = f"%-{components['suffix_base']}_RC%"
        else:
            # For non-RC testplan_ids, match the exact suffix
            suffix_pattern = f"%-{components['suffix']}"

        if debug:
            print(f"Is RC testplan_id: {components['is_rc']}")
            print(f"Suffix base: {components['suffix_base']}")
            print(f"Suffix pattern: {suffix_pattern}")

        # First, try to find a testplan_id with the same prefix, version, and suffix but a lower build number
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :suffix_pattern 
            AND testplan_id < :current_testplan_id 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC 
            LIMIT 1
        """
        )

        prefix_pattern = f"{components['prefix']}-v{components['version']}%"

        if debug:
            print(f"Query 1 - Looking for testplan_id with:")
            print(f"  prefix_pattern: {prefix_pattern}")
            print(f"  suffix_pattern: {suffix_pattern}")
            print(f"  current_testplan_id: {testplan_id}")

        result = session.execute(
            query,
            {"prefix_pattern": prefix_pattern, "suffix_pattern": suffix_pattern, "current_testplan_id": testplan_id},
        ).fetchone()

        if result:
            return result[0]

        # If no result found, try a more general query but still enforce the same suffix
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :suffix_pattern 
            AND testplan_id < :current_testplan_id 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC 
            LIMIT 1
        """
        )

        prefix_pattern = f"{components['prefix']}-%"
        # Suffix pattern is already set above

        if debug:
            print(f"Query 2 - Looking for testplan_id with:")
            print(f"  prefix_pattern: {prefix_pattern}")
            print(f"  suffix_pattern: {suffix_pattern}")
            print(f"  current_testplan_id: {testplan_id}")

        result = session.execute(
            query,
            {"prefix_pattern": prefix_pattern, "suffix_pattern": suffix_pattern, "current_testplan_id": testplan_id},
        ).fetchone()

        if result:
            return result[0]

        # If still no result found, try a query that allows for different version numbers
        query = text(
            """
            SELECT testplan_id FROM result 
            WHERE testplan_id LIKE :prefix_pattern 
            AND testplan_id LIKE :version_pattern
            AND testplan_id LIKE :suffix_pattern 
            AND testplan_id < :current_testplan_id 
            GROUP BY testplan_id
            ORDER BY testplan_id DESC 
            LIMIT 1
        """
        )

        prefix_pattern = f"{components['prefix']}-%"
        version_pattern = "%-v%"
        # Suffix pattern is already set above

        if debug:
            print(f"Query 3 - Looking for testplan_id with:")
            print(f"  prefix_pattern: {prefix_pattern}")
            print(f"  version_pattern: {version_pattern}")
            print(f"  suffix_pattern: {suffix_pattern}")
            print(f"  current_testplan_id: {testplan_id}")

        result = session.execute(
            query,
            {
                "prefix_pattern": prefix_pattern,
                "version_pattern": version_pattern,
                "suffix_pattern": suffix_pattern,
                "current_testplan_id": testplan_id,
            },
        ).fetchone()

        if result:
            return result[0]

        return None

    except Exception as e:
        print(f"Error querying database: {e}", file=sys.stderr)
        return None


def get_all_testplans_df(session, debug=False):
    """
    Query the database to get all distinct testplan_ids and return as a DataFrame.

    Args:
        session: SQLAlchemy session
        debug: Whether to print debug information

    Returns:
        A pandas DataFrame with all testplan_ids
    """
    try:
        if debug:
            print("Fetching all testplan_ids from database...")

        # Query to get all distinct testplan_ids
        query = text(
            """
            SELECT DISTINCT testplan_id FROM result
            ORDER BY testplan_id DESC
        """
        )

        # Execute the query and fetch all results
        results = session.execute(query).fetchall()

        # Create a DataFrame from the results
        df = pd.DataFrame(results, columns=["testplan_id"])

        if debug:
            print(f"Found {len(df)} distinct testplan_ids")

        return df

    except Exception as e:
        print(f"Error getting all testplan_ids: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["testplan_id"])


def get_previous_testplan_id_from_df(df, testplan_id, debug=False):
    """
    Find the previous testplan_id from a DataFrame of testplan_ids.

    Args:
        df: DataFrame containing all testplan_ids
        testplan_id: The current testplan_id
        debug: Whether to print debug information

    Returns:
        The previous testplan_id or None if not found
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        if debug:
            print(f"Looking for previous testplan_id for {testplan_id}")
            print(f"Components: {components}")

        # Filter the DataFrame for testplan_ids with the same prefix and version
        prefix = components["prefix"]
        version = components["version"]

        # First, try to find a testplan_id with the same prefix, version, and suffix
        if components["is_rc"]:
            # For RC testplan_ids, match the base suffix and any RC number
            suffix_base = components["suffix_base"]
            mask = (
                df["testplan_id"].str.startswith(f"{prefix}-v{version}")
                & df["testplan_id"].str.contains(f"-{suffix_base}_RC")
                & (df["testplan_id"] < testplan_id)
            )
        else:
            # For non-RC testplan_ids, check if it's a _win- format or standard format
            suffix = components["suffix"]
            if "_win" in components["build_number"]:
                # For _win- format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-v{version}")
                    & df["testplan_id"].str.contains(f"_win-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )
            else:
                # For standard format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-v{version}")
                    & df["testplan_id"].str.endswith(f"-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )

        filtered_df = df[mask]

        if not filtered_df.empty:
            # Get the first (most recent) testplan_id
            previous_testplan_id = filtered_df.iloc[0]["testplan_id"]
            if debug:
                print(f"Found previous testplan_id with same prefix, version, and suffix: {previous_testplan_id}")
            return previous_testplan_id

        # If no result found, try a more general query with just the same prefix and suffix
        if components["is_rc"]:
            # For RC testplan_ids, match the base suffix and any RC number
            suffix_base = components["suffix_base"]
            mask = (
                df["testplan_id"].str.startswith(f"{prefix}-")
                & df["testplan_id"].str.contains(f"-{suffix_base}_RC")
                & (df["testplan_id"] < testplan_id)
            )
        else:
            # For non-RC testplan_ids, check if it's a _win- format or standard format
            suffix = components["suffix"]
            if "_win" in components["build_number"]:
                # For _win- format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-")
                    & df["testplan_id"].str.contains(f"_win-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )
            else:
                # For standard format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-")
                    & df["testplan_id"].str.endswith(f"-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )

        filtered_df = df[mask]

        if not filtered_df.empty:
            # Get the first (most recent) testplan_id
            previous_testplan_id = filtered_df.iloc[0]["testplan_id"]
            if debug:
                print(f"Found previous testplan_id with same prefix and suffix: {previous_testplan_id}")
            return previous_testplan_id

        # If still no result found, try an even more general query
        if components["is_rc"]:
            # For RC testplan_ids, match the base suffix and any RC number
            suffix_base = components["suffix_base"]
            mask = (
                df["testplan_id"].str.startswith(f"{prefix}-")
                & df["testplan_id"].str.contains(f"-{suffix_base}_RC")
                & (df["testplan_id"] < testplan_id)
            )
        else:
            # For non-RC testplan_ids, check if it's a _win- format or standard format
            suffix = components["suffix"]
            if "_win" in components["build_number"]:
                # For _win- format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-")
                    & df["testplan_id"].str.contains("-v")
                    & df["testplan_id"].str.contains(f"_win-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )
            else:
                # For standard format testplan_ids
                mask = (
                    df["testplan_id"].str.startswith(f"{prefix}-")
                    & df["testplan_id"].str.contains("-v")
                    & df["testplan_id"].str.endswith(f"-{suffix}")
                    & (df["testplan_id"] < testplan_id)
                )

        filtered_df = df[mask]

        if not filtered_df.empty:
            # Get the first (most recent) testplan_id
            previous_testplan_id = filtered_df.iloc[0]["testplan_id"]
            if debug:
                print(f"Found previous testplan_id with general query: {previous_testplan_id}")
            return previous_testplan_id

        if debug:
            print(f"No previous testplan_id found for {testplan_id}")
        return None

    except Exception as e:
        print(f"Error finding previous testplan_id from DataFrame: {e}", file=sys.stderr)
        return None


def get_previous_release_testplan_id_from_df(df, testplan_id, debug=False):
    """
    Find the previous release testplan_id from a DataFrame of testplan_ids.

    Args:
        df: DataFrame containing all testplan_ids
        testplan_id: The current testplan_id
        debug: Whether to print debug information

    Returns:
        The previous release testplan_id or None if not found
    """
    try:
        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)

        # Extract the major version (e.g., 2.40 from 2.40.0)
        version_parts = components["version"].split(".")
        major_version = ".".join(version_parts[:2])  # e.g., "2.40"

        # Get the base suffix without any RC part
        base_suffix = components["suffix"]
        if "_RC" in base_suffix:
            base_suffix = base_suffix.split("_RC")[0]

        # Find all previous versions with RC testplans
        major_version_parts = major_version.split(".")
        major = int(major_version_parts[0])
        minor = int(major_version_parts[1])

        if debug:
            print(f"Looking for previous release testplan_id for {testplan_id}")
            print(f"Major version: {major_version}, Base suffix: {base_suffix}")

        # Filter the DataFrame for RC testplan_ids
        prefix = components["prefix"]
        mask = df["testplan_id"].str.startswith(f"{prefix}-") & df["testplan_id"].str.contains(f"-{base_suffix}_RC")

        filtered_df = df[mask]

        if filtered_df.empty:
            if debug:
                print(f"No RC testplans found")
            return None

        # Group testplans by version and find the highest RC for each version
        version_rc_testplans = {}

        for _, row in filtered_df.iterrows():
            testplan = row["testplan_id"]
            try:
                # Extract version
                version_match = re.search(r"-v([\d\.]+)", testplan)
                if version_match:
                    version = version_match.group(1)
                    version_parts = version.split(".")
                    if len(version_parts) >= 2:
                        ver_major = int(version_parts[0])
                        ver_minor = int(version_parts[1])

                        # Only consider versions less than the current version
                        if ver_major < major or (ver_major == major and ver_minor < minor):
                            # Extract RC number
                            rc_match = re.search(r"_RC(\d+)$", testplan)
                            if rc_match:
                                rc_num = int(rc_match.group(1))
                                ver_key = f"{ver_major}.{ver_minor}"

                                # Store the highest RC for each version
                                if ver_key not in version_rc_testplans or rc_num > version_rc_testplans[ver_key][0]:
                                    version_rc_testplans[ver_key] = (rc_num, testplan)
            except Exception as e:
                if debug:
                    print(f"Error parsing testplan {testplan}: {e}")
                continue

        if not version_rc_testplans:
            if debug:
                print(f"No valid RC testplans found for previous versions")
            return None

        # Sort versions in descending order
        sorted_versions = sorted(
            version_rc_testplans.keys(), key=lambda v: [int(x) for x in v.split(".")], reverse=True
        )

        if debug:
            print(f"Found RC testplans for versions: {sorted_versions}")
            for ver in sorted_versions:
                rc_num, testplan = version_rc_testplans[ver]
                print(f"  Version {ver}: RC{rc_num} - {testplan}")

        # Return the highest RC testplan from the most recent version
        if sorted_versions:
            latest_version = sorted_versions[0]
            rc_num, testplan = version_rc_testplans[latest_version]
            if debug:
                print(f"Selected version {latest_version} RC{rc_num}: {testplan}")
            return testplan

        return None

    except Exception as e:
        print(f"Error finding previous release testplan_id from DataFrame: {e}", file=sys.stderr)
        return None


def iterate_db_get_testplan(testplan_id):
    yaml_file = CONSOLIDATED_REPORTS.qa2_config_file_path

    if not yaml_file:
        print("Error: QA2_CONFIG_FILE_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        debug = False

        # Read the YAML config
        yaml_info = read_yaml(yaml_file)

        # Get MySQL connection info
        mysql_info = yaml_info["Database"]
        username, password, port, database, hostname = get_mysql_db_info(mysql_info)

        # Create database connection
        uri = f"mysql+pymysql://{username}:{quote_plus(password)}@{hostname}:{port}/{database}"
        engine = create_engine(uri)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Parse the testplan_id to get its components
        components = parse_testplan_id(testplan_id)
        if debug:
            print(f"Parsed testplan_id components: {components}")

        # Get all testplan_ids as a DataFrame
        df = get_all_testplans_df(session, debug)

        # Get the previous testplan_id using the DataFrame
        previous_testplan_id = get_previous_testplan_id_from_df(df, testplan_id, debug)

        # Get the previous release testplan_id using the DataFrame
        previous_release_testplan_id = get_previous_release_testplan_id_from_df(df, testplan_id, debug)

        # print(f"Current testplan_id: {testplan_id}")

        # Check if previous testplan_id has execution status >= 90%
        if previous_testplan_id:
            exec_status, p_n_df = get_testplan_execution_status_df(session, previous_testplan_id, debug)
            # print(f"Previous testplan_id: {previous_testplan_id}")

            if exec_status is not None and exec_status >= 90:
                # print(f"  Execution status: {exec_status}% (Valid - >= 90%)")
                pass
            else:
                # print(f"  Execution status: {exec_status}% (Invalid - < 90%)")
                pass
                # Find the next previous testplan_id with execution status >= 90%
                valid_previous_testplan_id = get_valid_previous_testplan_id(session, testplan_id, 90, debug)
                if valid_previous_testplan_id:
                    valid_exec_status, p_n_df = get_testplan_execution_status_df(
                        session, valid_previous_testplan_id, debug
                    )
                    print(
                        f"  Valid previous testplan_id: {valid_previous_testplan_id} (Execution status: {valid_exec_status}%)"
                    )
                else:
                    print("  No valid previous testplan_id found with execution status >= 90%")
        else:
            print(f"No previous testplan_id found for {testplan_id}")

        # Check if previous release testplan_id has execution status >= 90%
        if previous_release_testplan_id:
            exec_status, p_r_df = get_testplan_execution_status_df(session, previous_release_testplan_id, debug)
            # print(f"Previous release testplan_id: {previous_release_testplan_id}")

            if exec_status is not None and exec_status >= 90:
                # print(f"  Execution status: {exec_status}% (Valid - >= 90%)")
                pass
            else:
                # print(f"  Execution status: {exec_status}% (Invalid - < 90%)")
                # Find the next previous release testplan_id with execution status >= 90%
                valid_previous_release_testplan_id = get_valid_previous_release_testplan_id(
                    session, testplan_id, 90, debug
                )
                if valid_previous_release_testplan_id:
                    valid_exec_status, p_r_df = get_testplan_execution_status_df(
                        session, valid_previous_release_testplan_id, debug
                    )
                    print(
                        f"  Valid previous release testplan_id: {valid_previous_release_testplan_id} (Execution status: {valid_exec_status}%)"
                    )
                else:
                    print("  No valid previous release testplan_id found with execution status >= 90%")
        else:
            print(f"No previous release testplan_id found for {testplan_id}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if "session" in locals():
            session.close()
    return p_n_df, p_r_df, previous_testplan_id, previous_release_testplan_id

"""MySQL client for the Issue Grouping system.

Provides :class:`ConnectToMySql`, the single database access object used
throughout the pipeline for reading test results and writing cluster
assignments back to the ``error_map_qgenie`` table.

Layering
--------
This module sits in the **data** layer.  It imports only from:
- ``src.core`` (exceptions, interfaces)
- ``src.constants`` (DataFrameKeys)
- ``src.logger``
- Standard library / third-party packages

It must **not** import from ``src.helpers``, ``src.clustering``, or any
other higher-level package.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

import mysql.connector as msqlconnector
import pandas as pd

from src.constants import DataFrameKeys
from src.core.exceptions import DatabaseError
from src.core.interfaces import IDataLoader
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "ConnectToMySql",
    "get_sql_connection",
    "sql_connection",
    "get_tc_ids_from_sql",
    "update_error_map_qgenie_table",
    "get_error_group_id",
    "find_regressions_between_two_tests",
    "get_tc_id_df",
]


class ConnectToMySql(IDataLoader):
    """MySQL database client for the ``mlg-qa`` schema.

    Wraps all SQL interactions behind a clean Python API so that the rest of
    the codebase never constructs raw SQL strings or manages connections
    directly.

    All connections are opened on demand via :meth:`connection_context` and
    closed immediately after each operation — no persistent connection pool is
    maintained, which avoids stale-connection issues in long-running processes.

    Args:
        user: MySQL username.  Defaults to the shared read-write account.
        secret: MySQL password.
        host: Hostname of the MySQL server.
        db: Database / schema name.

    Example:
        >>> client = ConnectToMySql()
        >>> df = client.fetch_result_based_on_runid("QNN-v2.46.0.260319_nightly")
    """

    def __init__(
        self,
        user: str = "mlg_rw",
        secret: str = "gH@d8Jk9@1",
        host: str = "hydcrpmysqlprd10",
        db: str = "mlg-qa",
    ) -> None:
        self.user = user
        self.secret = secret
        self.host = host
        self.db = db

    # ------------------------------------------------------------------
    # IDataLoader contract
    # ------------------------------------------------------------------

    def load(self, **kwargs) -> pd.DataFrame:
        """Load test data by run ID (satisfies :class:`IDataLoader`).

        Args:
            **kwargs: Must contain ``run_id`` (str).

        Returns:
            DataFrame of test results for the given run ID.

        Raises:
            ValueError: If ``run_id`` is not provided.
            DatabaseError: On connection or query failures.
        """
        run_id = kwargs.get("run_id")
        if not run_id:
            raise ValueError("`run_id` is required for ConnectToMySql.load()")
        return self.fetch_result_based_on_runid(run_id)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        """Open a new MySQL connection.

        Returns:
            A ``mysql.connector`` connection object.

        Raises:
            DatabaseError: On authentication or connection failure.
        """
        try:
            return msqlconnector.connect(
                user=self.user,
                password=self.secret,
                host=self.host,
                database=self.db,
                use_pure=True,
            )
        except msqlconnector.Error as err:
            if err.errno == msqlconnector.errorcode.ER_ACCESS_DENIED_ERROR:
                raise DatabaseError("Authentication failed: check username and password", cause=err)
            if err.errno == msqlconnector.errorcode.ER_BAD_DB_ERROR:
                raise DatabaseError(f"Database '{self.db}' does not exist", cause=err)
            raise DatabaseError(f"MySQL connection error: {err}", cause=err)

    @contextmanager
    def connection_context(self) -> Generator:
        """Context manager that opens a connection and ensures it is closed.

        Yields:
            An active ``mysql.connector`` connection.

        Example:
            >>> with client.connection_context() as cnx:
            ...     df = pd.read_sql("SELECT 1", cnx)
        """
        connection = self.connect()
        try:
            yield connection
        finally:
            if connection:
                connection.close()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def fetch_data(self, query: str) -> pd.DataFrame:
        """Execute an arbitrary SQL query and return results as a DataFrame.

        Args:
            query: SQL query string.

        Returns:
            DataFrame of results; empty DataFrame if the query returns no rows.

        Raises:
            DatabaseError: On connection or query failure.
        """
        try:
            with self.connection_context() as cnx:
                logger.info(f"Executing query: {query}")
                df = pd.read_sql(query, cnx)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Query failed: {exc}", query=query, cause=exc) from exc

        if df.empty:
            logger.warning(f"Query returned no rows: {query}")
        return df

    def fetch_runids(self, filters: Optional[str] = None, fetch_all: bool = True) -> pd.DataFrame:
        """Fetch all testplan IDs from the result tables.

        Searches the current ``result`` table and up to 5 months of historical
        ``result_YYYY_MM`` tables.  Strips ``gerritsanity`` entries from
        results.

        Args:
            filters: Optional LIKE-clause fragment to narrow testplan IDs.
            fetch_all: When ``True`` (default), search current and historical
                result tables; when ``False``, only the current ``result`` table.

        Returns:
            DataFrame with a ``testplan_id`` column, sorted descending.
        """
        try:
            with self.connection_context() as cnx:
                tables = self._get_past_result_table_names(include_result=True) if fetch_all else ["result"]
                filter_clause = f"WHERE testplan_id LIKE '%{filters}%'" if filters else ""
                frames = []
                for table in tables:
                    query = f"SELECT DISTINCT(testplan_id) FROM {table} " f"{filter_clause} ORDER BY testplan_id DESC"
                    frames.append(pd.read_sql(query, cnx))
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError("Failed to fetch run IDs", cause=exc) from exc

        if not frames:
            return pd.DataFrame()

        overall = pd.concat(frames, ignore_index=True)
        if overall.empty:
            return pd.DataFrame()

        overall = overall.sort_values(by=["testplan_id"], ascending=False)
        return overall[~overall["testplan_id"].str.contains("gerritsanity", case=False, na=False)]

    def fetch_result_based_on_runid(self, runid: str) -> pd.DataFrame:
        """Fetch all test records for a given testplan ID.

        Searches current and historical result tables until data is found.

        Args:
            runid: Testplan ID to look up (e.g.
                ``"QNN-v2.46.0.260319041023_nightly"``).

        Returns:
            DataFrame of test records; empty DataFrame if not found in any table.
        """
        for table in self._get_past_result_table_names(include_result=True):
            try:
                with self.connection_context() as cnx:
                    logger.info(f"Checking result table: {table}")
                    df = pd.read_sql(f'SELECT * FROM {table} WHERE testplan_id = "{runid}";', cnx)
                if not df.empty:
                    return df
                logger.warning(f"No data for run_id={runid} in table={table}")
            except Exception as exc:
                logger.warning(f"Error querying {table} for {runid}: {exc}")
        return pd.DataFrame()

    def get_regressions(self, test_id_a: str, test_id_b: str) -> pd.DataFrame:
        """Fetch test-cases that regressed between two run IDs (PASS→FAIL).

        Automatically sorts the two IDs so the newer run is ``test_id_a``.

        Args:
            test_id_a: First run ID.
            test_id_b: Second run ID.

        Returns:
            DataFrame of regressed test cases with result, reason, scores, etc.
        """
        test_id_a, test_id_b = self.sort_run_ids(test_id_a, test_id_b)
        logger.info(f"Regression query: new={test_id_a}, old={test_id_b}")

        for table in self._get_past_result_table_names(include_result=True):
            try:
                query = f"""
                SELECT
                    r1.tc_uuid, r1.model_name AS name, r1.soc_name, r1.runtime,
                    r1.type, r1.result, r1.score, r1.reason, r1.tags,
                    r1.converter_options, r1.quantize_options, r1.inference_options,
                    r1.graph_prepare, r1.graph_execute, r1.jira_id,
                    r1.log AS log_path, r1.dsp_type
                FROM {table} r1
                JOIN {table} r2 ON r1.tc_uuid = r2.tc_uuid
                WHERE r1.testplan_id = "{test_id_a}"
                  AND r2.testplan_id = "{test_id_b}"
                  AND r1.result = 'FAIL'
                  AND r2.result = 'PASS'
                """
                with self.connection_context() as cnx:
                    df = pd.read_sql(query, cnx)
                if not df.empty:
                    return df
            except Exception as exc:
                logger.warning(f"Regression query failed for {table}: {exc}")

        return pd.DataFrame()

    def get_error_id_row(self, type: str, runtime: str, cluster_name: str) -> pd.DataFrame:
        """Fetch an error_map_qgenie row by type, runtime, and cluster name.

        Args:
            type: Test type (e.g. ``"quantizer"``).
            runtime: Runtime string (e.g. ``"htp_fp16"``).
            cluster_name: Lowercased cluster name.

        Returns:
            Single-row DataFrame, or empty DataFrame if not found.
        """
        query = (
            f"SELECT * FROM error_map_qgenie "
            f'WHERE test_type = "{type}" AND runtime = "{runtime}" '
            f'AND cluster_name = "{cluster_name}";'
        )
        df = self.fetch_data(query)
        if df.empty:
            logger.warning(f"No error_map_qgenie row for type={type}, runtime={runtime}, cluster={cluster_name}")
        return df

    def get_error_group_id(self, type: str, runtime: str, cluster_name: str) -> str:
        """Return the ``error_group_id`` for a given cluster, or empty string.

        Args:
            type: Test type.
            runtime: Runtime string.
            cluster_name: Cluster name.

        Returns:
            The ``error_group_id`` string, or ``""`` if not found.
        """
        query = """
            SELECT error_group_id FROM error_map_qgenie
            WHERE test_type = %s AND runtime = %s AND cluster_name = %s;
        """
        try:
            with self.connection_context() as cnx:
                cursor = cnx.cursor()
                cursor.execute(query, (type.lower(), runtime.lower(), cluster_name.lower()))
                result = cursor.fetchone()
                cursor.close()
        except Exception as exc:
            logger.error(f"get_error_group_id failed: {exc}")
            return ""

        return result[0] if result else ""

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def update_qgenie_error_map_table(self, input_df: pd.DataFrame) -> None:
        """Insert or timestamp-update clusters in the ``error_map_qgenie`` table.

        Inserts rows for (cluster_name, runtime, type) triples that do not
        yet exist; updates ``updatedAt`` for triples that do.

        Args:
            input_df: DataFrame that must contain columns: ``clusters``,
                ``runtime``, ``reason``, ``type``, ``cluster_class``.

        Raises:
            DatabaseError: On unexpected write failures.
        """
        required = [DataFrameKeys.cluster_name, "runtime", "reason", "type", DataFrameKeys.cluster_class]
        unique_on = [DataFrameKeys.cluster_name, "runtime", "type"]

        if not all(col in input_df.columns for col in required):
            logger.error(f"Missing required columns for error_map_qgenie update: {required}")
            return

        input_df = input_df.dropna(subset=required)
        unique_rows = input_df[required].drop_duplicates(subset=unique_on)
        if unique_rows.empty:
            logger.info("No valid cluster rows to persist.")
            return

        try:
            with self.connection_context() as cnx:
                existing = pd.read_sql("SELECT * FROM error_map_qgenie;", cnx)
                existing_pairs = set(zip(existing["cluster_name"], existing["runtime"], existing["test_type"]))
                cursor = cnx.cursor()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                new_rows, update_rows = [], []
                for _, row in unique_rows.iterrows():
                    cluster_name = row[DataFrameKeys.cluster_name].strip().lower()
                    runtime = row["runtime"].strip().lower()
                    reason = row["reason"].lower()
                    test_type = row["type"].lower()
                    cluster_class = row[DataFrameKeys.cluster_class].strip().lower()

                    if (cluster_name, runtime, test_type) not in existing_pairs:
                        new_rows.append(
                            (
                                cluster_name,
                                runtime,
                                reason,
                                test_type,
                                self._generate_key(f"{cluster_name}_{test_type}_{runtime}"),
                                cluster_class,
                                now,
                                now,
                            )
                        )
                    else:
                        update_rows.append((now, cluster_name, runtime, test_type))

                if new_rows:
                    cursor.executemany(
                        """INSERT INTO error_map_qgenie
                           (cluster_name, runtime, error_reason, test_type,
                            error_group_id, cluster_class, createdAt, updatedAt)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        new_rows,
                    )
                    logger.info(f"Inserted {len(new_rows)} new error_map_qgenie entries.")

                if update_rows:
                    cursor.executemany(
                        """UPDATE error_map_qgenie SET updatedAt = %s
                           WHERE cluster_name = %s AND runtime = %s AND test_type = %s""",
                        update_rows,
                    )
                    logger.info(f"Updated {len(update_rows)} error_map_qgenie timestamps.")

                cnx.commit()
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError("Failed to update error_map_qgenie", cause=exc) from exc

        time.sleep(5)  # brief pause to avoid overwhelming the DB on bulk runs

    def update_error_group_class(self, df: pd.DataFrame, cluster_class: str) -> None:
        """Update the ``cluster_class`` for rows in *df* that have no class yet.

        Args:
            df: DataFrame with ``type``, ``runtime``, and ``clusters`` columns.
            cluster_class: Class label to assign.
        """
        update_query = """
            UPDATE error_map_qgenie SET cluster_class = %s
            WHERE error_group_id = %s;
        """
        try:
            with self.connection_context() as cnx:
                cursor = cnx.cursor()
                for _, row in df.iterrows():
                    id_df = self.get_error_id_row(
                        type=row["type"],
                        runtime=row["runtime"],
                        cluster_name=row[DataFrameKeys.cluster_name].lower(),
                    )
                    if id_df.empty:
                        continue
                    error_group_id = id_df.iloc[0]["error_group_id"]
                    existing_class = id_df.iloc[0]["cluster_class"]
                    if error_group_id and existing_class is None:
                        cursor.execute(update_query, (cluster_class, error_group_id))
                cnx.commit()
        except Exception as exc:
            logger.error(f"update_error_group_class failed: {exc}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def sort_run_ids(self, run_id1: str, run_id2: str) -> tuple[str, str]:
        """Sort two run IDs so the newer one (by embedded timestamp) comes first.

        Supports both ``vX.Y.Z.YYYYMMDDHHMMSS`` and bare 12-digit timestamp
        formats embedded in the run ID string.

        Args:
            run_id1: First run ID string.
            run_id2: Second run ID string.

        Returns:
            A (newer, older) tuple.
        """
        main_pat = re.compile(r"v\d+\.\d+\.\d+\.([0-9]{12})(?=[_-]|$)")
        fallback_pat = re.compile(r"(\d{12})(?!\d)")

        def _ts(rid: str) -> int:
            m = main_pat.search(rid)
            if m:
                return int(m.group(1))
            last = None
            for last in fallback_pat.finditer(rid):
                pass
            return int(last.group(1)) if last else -1

        return (run_id1, run_id2) if _ts(run_id1) >= _ts(run_id2) else (run_id2, run_id1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_key(text: str) -> str:
        """Return a 10-character MD5 hex digest of *text*."""
        return hashlib.md5(text.encode()).hexdigest()[:10]

    def _get_past_result_table_names(
        self,
        num_of_months_to_check: int = 5,
        include_result: bool = False,
    ) -> list[str]:
        """Return result table names to check (current + historical).

        Args:
            num_of_months_to_check: How many months back to look.
            include_result: Prepend the bare ``result`` table when ``True``.

        Returns:
            Ordered list of table name strings.
        """
        now = datetime.now()
        year, month = now.year, now.month
        tables = []
        if include_result:
            tables.append("result")
        for back in range(1, num_of_months_to_check):
            m = month - back
            y = year
            if m <= 0:
                m += 12
                y -= 1
            tables.append(f"result_{y}_0{m}" if m <= 9 else f"result_{y}_{m}")
        return tables


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

parquet_file = "run_ids.parquet"

_sql_connection: ConnectToMySql | None = None


def get_sql_connection() -> ConnectToMySql:
    """Return the process-level MySQL singleton, creating it on first call."""
    global _sql_connection
    if _sql_connection is None:
        _sql_connection = ConnectToMySql()
    return _sql_connection


# Module-level singleton for backward compatibility (matches helpers.py pattern)
sql_connection = ConnectToMySql()


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


@execution_timer
def get_tc_ids_from_sql():
    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        return df
    else:
        run_ids = sql_connection.fetch_runids()
        run_ids.to_parquet(parquet_file)
        return run_ids


@execution_timer
def update_error_map_qgenie_table(df):
    try:
        sql_connection.update_qgenie_error_map_table(df)
    except Exception as e:
        logger.error(f"Failed to update SQL Table: {e}")


@execution_timer
def get_error_group_id(error_type: str, runtime: str, cluster_name: str) -> str:
    return sql_connection.get_error_group_id(error_type, runtime, cluster_name)


@execution_timer
def find_regressions_between_two_tests(tc_id_a: str, tc_id_b: str) -> pd.DataFrame:
    return sql_connection.get_regressions(tc_id_a, tc_id_b)


@execution_timer
def get_tc_id_df(tc_id: str):
    return sql_connection.fetch_result_based_on_runid(tc_id)

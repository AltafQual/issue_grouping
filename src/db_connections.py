import hashlib
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import mysql.connector as msqlconnector
import pandas as pd

from src.constants import DataFrameKeys

logger = logging.getLogger(__name__)


class DatabaseConnection(ABC):
    """
    Abstract base class for database connections.
    Provides a common interface for different database types.
    """

    @abstractmethod
    def connect(self):
        """
        Establish a connection to the database.

        Returns:
            Connection object specific to the database type
        """
        pass

    @abstractmethod
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the results as a DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame containing query results
        """
        pass

    @contextmanager
    def connection_context(self):
        """
        Context manager for database connections.
        Ensures connections are properly closed after use.

        Yields:
            Active database connection
        """
        connection = self.connect()
        try:
            yield connection
        finally:
            if connection:
                connection.close()


class ConnectToMySql(DatabaseConnection):
    """
    MySQL database connection handler.
    Provides methods to connect to MySQL and execute queries.
    """

    def __init__(
        self, user: str = "mlg_rw", secret: str = "gH@d8Jk9@1", host: str = "hydcrpmysqlprd10", db: str = "mlg-qa"
    ):
        """
        Initialize MySQL connection parameters.

        Args:
            user: MySQL username
            secret: MySQL password
            host: MySQL server hostname
            db: Database name
        """
        self.user = user
        self.secret = secret
        self.host = host
        self.db = db

    def connect(self):
        """
        Establish connection to MySQL database.

        Returns:
            MySQL connection object

        Raises:
            MySQL connector errors for connection issues
        """

        try:
            cnx = msqlconnector.connect(
                user=self.user, password=self.secret, host=self.host, database=self.db, use_pure=True
            )
            return cnx
        except msqlconnector.Error as err:
            if err.errno == msqlconnector.errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error("Authentication failed: Check username and password")
                raise
            elif err.errno == msqlconnector.errorcode.ER_BAD_DB_ERROR:
                logger.error(f"Database '{self.db}' does not exist")
                raise
            else:
                logger.error(f"Connection error: {err}")
                raise

    def fetch_runids(self, filters: Optional[str] = None, fetch_all: bool = True) -> pd.DataFrame:
        """
        Fetch test plan IDs from the database based on filters.

        Args:
            filters: String to filter testplan_ids (used in LIKE clause)
            fetch_all: If True, search in all result tables, otherwise only in the current result table

        Returns:
            DataFrame containing testplan_ids sorted in descending order
        """
        with self.connection_context() as cnx:
            if fetch_all:
                query = """
                SELECT table_name as TABLE_NAME
                FROM information_schema.tables
                    WHERE table_schema = 'mlg-qa' and
                        table_name like "result"
                        order by table_name desc limit 2;
                """
                table_df = pd.read_sql(query, cnx)
                tables = table_df["TABLE_NAME"].tolist()
                tables.append("result")

                overall_df = pd.DataFrame()
                for table_name in tables:
                    filter_clause = f"WHERE testplan_id LIKE '%{filters}%'" if filters else ""
                    query = f"""
                    SELECT DISTINCT(testplan_id) 
                    FROM {table_name} 
                    {filter_clause}
                    ORDER BY testplan_id DESC
                    """
                df = pd.read_sql(query, cnx)
                overall_df = pd.concat([overall_df, df], ignore_index=True)
            else:
                filter_clause = f"WHERE testplan_id LIKE '%{filters}%'" if filters else ""
                query = f"""
                SELECT DISTINCT(testplan_id) 
                FROM result 
                {filter_clause}
                ORDER BY testplan_id DESC
                """
                logger.info(query)
                overall_df = pd.read_sql(query, cnx)
        if overall_df.empty:
            return pd.DataFrame()

        tc_ids = overall_df.sort_values(by=["testplan_id"], ascending=False)
        return tc_ids[~tc_ids["testplan_id"].str.contains("gerritsanity", case=False, na=False)]

    def fetch_result_based_on_runid(self, runid: str) -> pd.DataFrame:
        """
        Fetch test plan results from the database based on testplan_id.

        Args:
            runid: Testplan ID to filter results

        Returns:
            DataFrame containing test plan results
        """
        with self.connection_context() as cnx:
            query = f'select * from result where testplan_id = "{runid}";'
            df = pd.read_sql(query, cnx)
            if df.empty:
                logger.info(f"No Data Found for runid: {runid}")
                return pd.DataFrame()
            return df

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.
        Args:
            query: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        return self._fetch_query_data(query)

    def _fetch_query_data(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the results as a DataFrame.

        Args:
            query: SQL query string to execute

        Returns:
            DataFrame containing query results
        """
        with self.connection_context() as cnx:
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, cnx)

        if df.empty:
            logger.error(f"No data found for query: {query}")
            return pd.DataFrame()

        return df

    def generate_key_from_testcase(self, testcase_str: str):
        return hashlib.md5(testcase_str.encode()).hexdigest()[:10]

    def update_qgenie_error_map_table(self, input_df: pd.DataFrame) -> None:
        """
        Insert unique (cluster_name, runtime) pairs from input_df into error_map_qgenie table
        if they don't already exist. Also assigns unique error_group_id and updates timestamps.

        Args:
            input_df: DataFrame containing cluster_name, runtime, reason, and type columns
        """
        required_columns = [DataFrameKeys.cluster_name, "runtime", "reason", "type", DataFrameKeys.cluster_class]
        unique_columns_subset = [DataFrameKeys.cluster_name, "runtime", "type"]
        if not all(col in input_df.columns for col in required_columns):
            logger.error(f"Input DataFrame must contain columns: {required_columns}")
            return

        input_df = input_df.dropna(subset=required_columns)
        unique_rows = input_df[required_columns].drop_duplicates(subset=unique_columns_subset)

        if unique_rows.empty:
            logger.info("No valid cluster_name/runtime rows found in input DataFrame.")
            return

        with self.connection_context() as cnx:
            # Fetch existing cluster_name/runtime pairs
            existing_query = "SELECT * FROM error_map_qgenie;"
            existing_df = pd.read_sql(existing_query, cnx)
            existing_pairs = set(zip(existing_df["cluster_name"], existing_df["runtime"], existing_df["test_type"]))
            cursor = cnx.cursor()

            # Prepare new rows and update existing ones
            new_rows = []
            update_rows = []
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for _, row in unique_rows.iterrows():
                try:
                    cluster_name = row[DataFrameKeys.cluster_name].strip().lower()
                    runtime = row["runtime"].strip().lower()
                    reason = row["reason"].lower()
                    test_type = row["type"].lower()
                    cluster_class = row[DataFrameKeys.cluster_class].strip().lower()

                    if (cluster_name, runtime, test_type) not in existing_pairs:
                        logger.info(f"Pushing row to sql: {row}")
                        new_rows.append(
                            (
                                cluster_name,
                                runtime,
                                reason,
                                test_type,
                                self.generate_key_from_testcase(f"{cluster_name}_{test_type}_{runtime}"),
                                cluster_class,
                                now,
                                now,
                            )
                        )
                    else:
                        update_rows.append((now, cluster_name, runtime, test_type))
                except Exception as e:
                    raise Exception(f"Error: {e} while processing row: {row}")

            # Insert new rows
            if new_rows:
                insert_query = """
                    INSERT INTO error_map_qgenie (cluster_name, runtime, error_reason, test_type, error_group_id, cluster_class, createdAt, updatedAt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                cursor.executemany(insert_query, new_rows)
                logger.info(f"Inserted {len(new_rows)} new entries into error_map_qgenie.")

            # Update existing rows' updatedAt
            if update_rows:
                update_query = """
                    UPDATE error_map_qgenie
                    SET updatedAt = %s
                    WHERE cluster_name = %s AND runtime = %s AND test_type = %s;
                """
                cursor.executemany(update_query, update_rows)
                logger.info(f"Updated {len(update_rows)} existing entries in error_map_qgenie.")

            cnx.commit()
            cnx.close()
            time.sleep(5)

    def update_error_group_class(self, df, cluster_class: str):
        update_query = """
            UPDATE error_map_qgenie
            SET cluster_class = %s
            WHERE error_group_id = %s;
        """

        with self.connection_context() as cnx:
            cursor = cnx.cursor()

            for _, row in df.iterrows():
                error_group_id_df = self.get_error_id_row(
                    type=row["type"], runtime=row["runtime"], cluster_name=row[DataFrameKeys.cluster_name].lower()
                )

                error_group_id, existing_cluster_class = None, None
                if not error_group_id_df.empty:
                    error_group_id = error_group_id_df.iloc[0]["error_group_id"]
                    existing_cluster_class = error_group_id_df.iloc[0]["cluster_class"]

                logger.info(f"error group id: {error_group_id}, exisitng class: {existing_cluster_class}")
                if error_group_id and existing_cluster_class is None:
                    cursor.execute(update_query, (cluster_class, error_group_id))

            cnx.commit()
            cnx.close()

    def get_error_id_row(self, type, runtime, cluster_name):
        query = f"""
            SELECT * FROM error_map_qgenie
            WHERE test_type = "{type}" AND runtime = "{runtime}" AND cluster_name = "{cluster_name}";
        """

        with self.connection_context() as cnx:
            logger.info(f"Executing query: {query}")
            df = pd.read_sql(query, cnx)

        if df.empty:
            logger.error(f"No data found for query: {query}")
            return pd.DataFrame()

        return df

    def get_error_group_id(self, type: str, runtime: str, cluster_name: str) -> str:
        query = """
            SELECT error_group_id FROM error_map_qgenie
            WHERE test_type = %s AND runtime = %s AND cluster_name = %s;
        """
        with self.connection_context() as cnx:
            cursor = cnx.cursor()
            cursor.execute(query, (type.lower(), runtime.lower(), cluster_name.lower()))
            result = cursor.fetchone()
            cursor.close()

        logger.info(f"Fetched result {result} with the details: {type}, {runtime}, {cluster_name}")
        return result[0] if result else ""

    def get_regressions(self, test_id_a: str, test_id_b: str):
        query = f"""
        SELECT 
            r1.tc_uuid AS tc_uuid,
            r1.model_name AS name,
            r1.soc_name AS soc_name,
            r1.runtime AS runtime,
            r1.type AS type,
            r1.result AS result,
            r1.score AS score,
            r1.reason AS reason,
            r1.tags as tags,
            r1.converter_options as converter_options,
            r1.quantize_options as quantize_options,
            r1.inference_options as inference_options,
            r1.graph_prepare as graph_prepare,
            r1.graph_execute as graph_execute,
            r1.jira_id as jira_id
                           
        FROM result r1
        JOIN result r2 ON r1.tc_uuid = r2.tc_uuid
        WHERE r1.testplan_id = "{test_id_a}"
        AND r2.testplan_id = "{test_id_b}"
        AND r1.result = 'FAIL'
        AND r2.result ='PASS'
        """
        with self.connection_context() as cnx:
            df = pd.read_sql(query, cnx)
        return df

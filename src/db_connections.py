import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional

import mysql.connector as msqlconnector
import pandas as pd

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
            query = f'select * from result where testplan_id = "{runid}" and result not in ("PASS");'
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

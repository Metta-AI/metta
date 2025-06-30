"""Database query logging utilities for performance monitoring."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator, Tuple

from psycopg import Connection, Cursor
from psycopg.abc import Query
from psycopg.sql import Composable

# Logger for database query performance
query_logger = logging.getLogger("db_performance")
query_logger.setLevel(logging.INFO)

# Threshold for slow query warnings (1 second)
SLOW_QUERY_THRESHOLD_SECONDS = 1.0


@contextmanager
def timed_query(
    con: Connection, query: Query, params: Tuple[Any, ...] = (), description: str = ""
) -> Generator[Cursor, None, None]:
    """
    Context manager that executes a database query with timing and logging.

    Logs all queries with their execution time. If a query takes longer than
    SLOW_QUERY_THRESHOLD_SECONDS, logs a warning with the query and parameters.

    Args:
        con: Database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Yields:
        Cursor with the executed query results
    """
    start_time = time.time()
    query_id = f"{description} " if description else ""

    try:
        with con.cursor() as cursor:
            cursor.execute(query, params)
            execution_time = time.time() - start_time

            # Log all queries with timing
            query_logger.info(f"{query_id}query completed in {execution_time:.3f}s")

            # Log slow queries with details
            if execution_time > SLOW_QUERY_THRESHOLD_SECONDS:
                query_str = query.as_string(con) if isinstance(query, Composable) else query.__str__()
                query_logger.warning(
                    f"SLOW QUERY ({execution_time:.3f}s): {query_id}\nQuery: {query_str}\nParams: {params}"
                )

            yield cursor

    except Exception as e:
        execution_time = time.time() - start_time
        query_logger.error(f"{query_id}query failed after {execution_time:.3f}s: {e}\nQuery: {query}\nParams: {params}")
        raise


def log_query_execution(con: Connection, query: Query, params: Tuple[Any, ...] = (), description: str = "") -> Any:
    """
    Execute a query with timing and logging, returning the results.

    This is a convenience function for simple queries that just need results.

    Args:
        con: Database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Returns:
        Query results (fetchall())
    """
    with timed_query(con, query, params, description) as cursor:
        return cursor.fetchall()


def log_query_execution_one(con: Connection, query: Query, params: Tuple[Any, ...] = (), description: str = "") -> Any:
    """
    Execute a query with timing and logging, returning the first result.

    Args:
        con: Database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Returns:
        First query result (fetchone())
    """
    with timed_query(con, query, params, description) as cursor:
        return cursor.fetchone()

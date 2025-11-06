"""Database query logging utilities for performance monitoring."""

import contextlib
import logging
import time
import typing

import psycopg
import psycopg.abc
import psycopg.rows
import psycopg.sql

# Logger for database query performance
query_logger = logging.getLogger("db_performance")
query_logger.setLevel(logging.INFO)

# Threshold for slow query warnings (1 second)
SLOW_QUERY_THRESHOLD_SECONDS = 1.0


@contextlib.asynccontextmanager
async def timed_query(
    con: psycopg.AsyncConnection,
    query: psycopg.abc.Query,
    params: typing.Tuple[typing.Any, ...] = (),
    description: str = "",
) -> typing.AsyncGenerator[psycopg.AsyncCursor, None]:
    """
    Async context manager that executes a database query with timing and logging.

    Args:
        con: Async database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Yields:
        AsyncCursor with the executed query results
    """
    start_time = time.time()
    query_id = f"{description} " if description else ""

    try:
        async with con.cursor() as cursor:
            await cursor.execute(query, params)
            execution_time = time.time() - start_time

            # Log all queries with timing
            query_logger.info(f"{query_id}query completed in {execution_time:.3f}s")

            # Log slow queries with details
            if execution_time > SLOW_QUERY_THRESHOLD_SECONDS:
                query_str = query.as_string(con) if isinstance(query, psycopg.sql.Composable) else query.__str__()
                query_logger.warning(
                    f"SLOW QUERY ({execution_time:.3f}s): {query_id}\nQuery: {query_str}\nParams: {params}"
                )

            yield cursor

    except Exception as e:
        execution_time = time.time() - start_time
        query_logger.error(f"{query_id}query failed after {execution_time:.3f}s: {e}\nQuery: {query}\nParams: {params}")
        raise


async def execute_query_and_log(
    con: psycopg.AsyncConnection,
    query: psycopg.abc.Query,
    params: typing.Tuple[typing.Any, ...] = (),
    description: str = "",
) -> list[psycopg.rows.TupleRow]:
    """
    Execute an async query with timing and logging, returning the results.

    Args:
        con: Async database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Returns:
        Query results (fetchall())
    """
    async with timed_query(con, query, params, description) as cursor:
        return await cursor.fetchall()


async def execute_single_row_query_and_log(
    con: psycopg.AsyncConnection,
    query: psycopg.abc.Query,
    params: typing.Tuple[typing.Any, ...] = (),
    description: str = "",
) -> psycopg.rows.TupleRow | None:
    """
    Execute an async query with timing and logging, returning the first result.

    Args:
        con: Async database connection
        query: SQL query string
        params: Query parameters
        description: Optional description for the query

    Returns:
        First query result (fetchone())
    """
    async with timed_query(con, query, params, description) as cursor:
        return await cursor.fetchone()

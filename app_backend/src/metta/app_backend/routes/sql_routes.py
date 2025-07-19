"""SQL query routes for self-service database access."""

import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from psycopg import errors as pg_errors
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.query_logger import execute_query_and_log
from metta.app_backend.route_logger import timed_route


class SQLQueryRequest(BaseModel):
    query: str


class SQLQueryResponse(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    row_count: int


class TableInfo(BaseModel):
    table_name: str
    column_count: int
    row_count: int


class TableSchema(BaseModel):
    table_name: str
    columns: List[Dict[str, Any]]


def create_sql_router(metta_repo: MettaRepo) -> APIRouter:
    """Create SQL query router with the provided MettaRepo instance."""
    router = APIRouter(prefix="/sql", tags=["sql"])
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.get("/tables", response_model=List[TableInfo])
    @timed_route("list_tables")
    async def list_tables(user: str = user_or_token) -> List[TableInfo]:
        """List all available tables in the database (excluding migrations)."""
        try:
            async with metta_repo.connect() as con:
                # Get all tables except schema_migrations
                tables_query = """
                    SELECT
                        t.table_name,
                        COUNT(c.column_name) as column_count
                    FROM information_schema.tables t
                    LEFT JOIN information_schema.columns c
                        ON t.table_name = c.table_name
                        AND t.table_schema = c.table_schema
                    WHERE t.table_schema = 'public'
                        AND t.table_type = 'BASE TABLE'
                        AND t.table_name != 'schema_migrations'
                    GROUP BY t.table_name
                    ORDER BY t.table_name
                """

                tables = await execute_query_and_log(con, tables_query, (), "list_tables_metadata")

                # Get row counts for each table
                table_info = []
                for table_name, column_count in tables:
                    row_count_query = "SELECT reltuples::bigint AS estimate FROM pg_class where relname = %s"
                    row_count_result = await execute_query_and_log(
                        con, row_count_query, (table_name,), f"count_rows_{table_name}"
                    )
                    row_count = row_count_result[0][0]

                    table_info.append(TableInfo(table_name=table_name, column_count=column_count, row_count=row_count))

                return table_info

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}") from e

    @router.get("/tables/{table_name}/schema", response_model=TableSchema)
    @timed_route("get_table_schema")
    async def get_table_schema(table_name: str, user: str = user_or_token) -> TableSchema:
        """Get the schema for a specific table."""
        try:
            async with metta_repo.connect() as con:
                # Verify table exists and is not schema_migrations
                if table_name == "schema_migrations":
                    raise HTTPException(status_code=403, detail="Access to schema_migrations table is not allowed")

                # Get column information
                schema_query = """
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                        AND table_name = %s
                    ORDER BY ordinal_position
                """

                columns = await execute_query_and_log(con, schema_query, (table_name,), f"get_schema_{table_name}")

                if not columns:
                    raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

                column_info = []
                for col in columns:
                    column_info.append(
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == "YES",
                            "default": col[3],
                            "max_length": col[4],
                        }
                    )

                return TableSchema(table_name=table_name, columns=column_info)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting table schema: {str(e)}") from e

    @router.post("/query", response_model=SQLQueryResponse)
    @timed_route("execute_sql_query")
    async def execute_query(request: SQLQueryRequest, user: str = user_or_token) -> SQLQueryResponse:
        """Execute a SQL query with a 20-second timeout."""
        try:
            # Basic validation to prevent access to schema_migrations
            query_lower = request.query.lower()
            if "schema_migrations" in query_lower:
                raise HTTPException(status_code=403, detail="Access to schema_migrations table is not allowed")

            # Ensure query is read-only (no writes allowed)
            # Check for common write operations
            write_keywords = ["insert", "update", "delete", "drop", "create", "alter", "truncate", "grant", "revoke"]
            first_word = query_lower.strip().split()[0] if query_lower.strip() else ""
            if first_word in write_keywords:
                raise HTTPException(
                    status_code=403, detail="Only read-only queries are allowed. Write operations are not permitted."
                )

            async def run_query():
                async with metta_repo.connect() as con:
                    # Set statement timeout to 20 seconds
                    await con.execute("SET statement_timeout = '20s'")

                    # Execute the query
                    result = await con.execute(request.query)

                    # Get column names
                    columns = []
                    if result.description:
                        columns = [desc.name for desc in result.description]

                    # Fetch all rows with a limit of 1000
                    rows = []
                    if result.rowcount > 0 or (result.rowcount == -1 and result.description):
                        rows = await result.fetchmany(1000)  # Limit to 1000 rows

                    # Convert rows to list of lists for JSON serialization
                    rows_list = [list(row) for row in rows]

                    return SQLQueryResponse(columns=columns, rows=rows_list, row_count=len(rows_list))

            # Run with asyncio timeout as additional safeguard
            return await asyncio.wait_for(run_query(), timeout=21.0)

        except asyncio.TimeoutError as e:
            raise HTTPException(status_code=408, detail="Query execution timed out after 20 seconds") from e
        except pg_errors.QueryCanceled as e:
            raise HTTPException(status_code=408, detail="Query execution timed out after 20 seconds") from e
        except pg_errors.SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"SQL syntax error: {str(e)}") from e
        except pg_errors.UndefinedTable as e:
            raise HTTPException(status_code=400, detail=f"Table not found: {str(e)}") from e
        except pg_errors.UndefinedColumn as e:
            raise HTTPException(status_code=400, detail=f"Column not found: {str(e)}") from e
        except pg_errors.InsufficientPrivilege as e:
            raise HTTPException(status_code=403, detail=f"Insufficient privileges: {str(e)}") from e

        except HTTPException:
            raise
        except Exception as e:
            # Log the full error for debugging but return a generic message
            error_type = type(e).__name__
            raise HTTPException(status_code=500, detail=f"Query execution failed ({error_type}): {str(e)}") from e

    return router

"""SQL query routes for self-service database access."""

import asyncio
import typing

import fastapi
import httpx
import psycopg
import pydantic

import metta.app_backend.auth
import metta.app_backend.config
import metta.app_backend.metta_repo
import metta.app_backend.query_logger
import metta.app_backend.route_logger


class SQLQueryRequest(pydantic.BaseModel):
    query: str


class SQLQueryResponse(pydantic.BaseModel):
    columns: typing.List[str]
    rows: typing.List[typing.List[typing.Any]]
    row_count: int


class TableInfo(pydantic.BaseModel):
    table_name: str
    column_count: int
    row_count: int


class TableSchema(pydantic.BaseModel):
    table_name: str
    columns: typing.List[typing.Dict[str, typing.Any]]


class AIQueryRequest(pydantic.BaseModel):
    description: str


class AIQueryResponse(pydantic.BaseModel):
    query: str


def create_sql_router(metta_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    """Create SQL query router with the provided MettaRepo instance."""
    router = fastapi.APIRouter(prefix="/sql", tags=["sql"])
    user_or_token = fastapi.Depends(metta.app_backend.auth.create_user_or_token_dependency(metta_repo))

    @router.get("/tables", response_model=typing.List[TableInfo])
    @metta.app_backend.route_logger.timed_route("list_tables")
    async def list_tables(user: str = user_or_token) -> typing.List[TableInfo]:
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

                tables = await metta.app_backend.query_logger.execute_query_and_log(
                    con, tables_query, (), "list_tables_metadata"
                )

                # Get row counts for each table
                table_info = []
                for table_name, column_count in tables:
                    row_count_query = "SELECT reltuples::bigint AS estimate FROM pg_class where relname = %s"
                    row_count_result = await metta.app_backend.query_logger.execute_query_and_log(
                        con, row_count_query, (table_name,), f"count_rows_{table_name}"
                    )
                    row_count = row_count_result[0][0]

                    table_info.append(TableInfo(table_name=table_name, column_count=column_count, row_count=row_count))

                return table_info

        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}") from e

    @router.get("/tables/{table_name}/schema", response_model=TableSchema)
    @metta.app_backend.route_logger.timed_route("get_table_schema")
    async def get_table_schema(table_name: str, user: str = user_or_token) -> TableSchema:
        """Get the schema for a specific table."""
        try:
            async with metta_repo.connect() as con:
                # Verify table exists and is not schema_migrations
                if table_name == "schema_migrations":
                    raise fastapi.HTTPException(
                        status_code=403, detail="Access to schema_migrations table is not allowed"
                    )

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

                columns = await metta.app_backend.query_logger.execute_query_and_log(
                    con, schema_query, (table_name,), f"get_schema_{table_name}"
                )

                if not columns:
                    raise fastapi.HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

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

        except fastapi.HTTPException:
            raise
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Error getting table schema: {str(e)}") from e

    @router.post("/query", response_model=SQLQueryResponse)
    @metta.app_backend.route_logger.timed_route("execute_sql_query")
    async def execute_query(request: SQLQueryRequest, user: str = user_or_token) -> SQLQueryResponse:
        """Execute a SQL query with a 20-second timeout."""
        try:
            # Basic validation to prevent access to schema_migrations
            query_lower = request.query.lower()
            if "schema_migrations" in query_lower:
                raise fastapi.HTTPException(status_code=403, detail="Access to schema_migrations table is not allowed")

            # Ensure query is read-only (no writes allowed)
            # Check for common write operations
            write_keywords = ["insert", "update", "delete", "drop", "create", "alter", "truncate", "grant", "revoke"]
            first_word = query_lower.strip().split()[0] if query_lower.strip() else ""
            if first_word in write_keywords:
                raise fastapi.HTTPException(
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
            raise fastapi.HTTPException(status_code=408, detail="Query execution timed out after 20 seconds") from e
        except psycopg.errors.QueryCanceled as e:
            raise fastapi.HTTPException(status_code=408, detail="Query execution timed out after 20 seconds") from e
        except psycopg.errors.SyntaxError as e:
            raise fastapi.HTTPException(status_code=400, detail=f"SQL syntax error: {str(e)}") from e
        except psycopg.errors.UndefinedTable as e:
            raise fastapi.HTTPException(status_code=400, detail=f"Table not found: {str(e)}") from e
        except psycopg.errors.UndefinedColumn as e:
            raise fastapi.HTTPException(status_code=400, detail=f"Column not found: {str(e)}") from e
        except psycopg.errors.InsufficientPrivilege as e:
            raise fastapi.HTTPException(status_code=403, detail=f"Insufficient privileges: {str(e)}") from e

        except fastapi.HTTPException:
            raise
        except Exception as e:
            # Log the full error for debugging but return a generic message
            error_type = type(e).__name__
            raise fastapi.HTTPException(
                status_code=500, detail=f"Query execution failed ({error_type}): {str(e)}"
            ) from e

    @router.post("/generate-query", response_model=AIQueryResponse)
    @metta.app_backend.route_logger.timed_route("generate_ai_query")
    async def generate_ai_query(request: AIQueryRequest, user: str = user_or_token) -> AIQueryResponse:
        """Generate a SQL query from natural language description using Claude."""
        # Get API key from environment variable
        if not metta.app_backend.config.anthropic_api_key:
            raise fastapi.HTTPException(status_code=500, detail="ANTHROPIC_API_KEY environment variable not set")

        # Fetch all table schemas in parallel
        tables = await list_tables(user)
        schemas = await asyncio.gather(*[get_table_schema(table.table_name, user) for table in tables])

        # Build schema description
        schema_lines = []
        for schema in schemas:
            schema_lines.append(f"Table: {schema.table_name}")
            for col in schema.columns:
                col_desc = f"    {col['name']} {col['type']}"
                if not col["nullable"]:
                    col_desc += " NOT NULL"
                if col["default"]:
                    col_desc += f" DEFAULT {col['default']}"
                schema_lines.append(col_desc)
            schema_lines.append("")  # Empty line between tables

        schema_description = "\n".join(schema_lines)

        prompt = (
            f"You are a SQL query generator. Given the following database schema "
            f"and a user's description, generate a SQL query that answers their request.\n\n"
            f"Database Schema:\n{schema_description}\n"
            f"User's request: {request.description}\n\n"
            f"Please respond with ONLY the SQL query, no explanation or markdown. "
            f"The query should be ready to execute."
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": metta.app_backend.config.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-opus-4-20250514",
                        "max_tokens": 1000,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
            data = response.json()
            generated_query = data["content"][0]["text"].strip()
            return AIQueryResponse(query=generated_query)

        except httpx.TimeoutException as e:
            raise fastapi.HTTPException(status_code=408, detail="Request to Claude API timed out") from e
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_msg = error_data.get("error", {}).get("message", f"API request failed: {e.response.status_code}")
            raise fastapi.HTTPException(status_code=e.response.status_code, detail=error_msg) from e
        except httpx.RequestError as e:
            raise fastapi.HTTPException(status_code=503, detail=f"Failed to connect to Claude API: {str(e)}") from e
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to generate query: {str(e)}") from e

    return router

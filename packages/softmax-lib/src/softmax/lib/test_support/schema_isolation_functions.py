import uuid
from urllib.parse import quote, urlparse, urlunparse

import psycopg


def create_isolated_schema_uri(base_uri: str, schema_name: str) -> str:
    """Create a database URI with a specific schema in the search path."""
    parsed = urlparse(base_uri)

    # Extract existing query parameters
    query_params = dict(param.split("=", 1) for param in parsed.query.split("&") if param)

    # Add or update the search_path option - URL encode the value
    # Include public schema so UUID functions are available
    query_params["options"] = quote(f"-csearch_path={schema_name},public")

    # Reconstruct the query string
    query = "&".join(f"{k}={v}" for k, v in query_params.items())

    # Build the new URI with the schema option
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))


def isolated_test_schema_uri(base_uri: str) -> str:
    """Create an isolated schema for testing.

    Returns the database URI configured to use the isolated schema.
    """
    # Generate a unique schema name for this test
    schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"

    # Create connection to create the schema
    with psycopg.connect(base_uri) as conn:
        with conn.cursor() as cursor:
            # Create the isolated schema
            cursor.execute(f"CREATE SCHEMA {schema_name}")
            # Ensure UUID extension is available (create it in public schema if not exists)
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            # Set search path to use our schema
            cursor.execute(f"SET search_path TO {schema_name}, public")
        conn.commit()

    # Create URI that uses the isolated schema
    schema_uri = create_isolated_schema_uri(base_uri, schema_name)

    return schema_uri

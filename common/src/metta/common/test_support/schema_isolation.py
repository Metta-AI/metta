"""Helpers for spinning up isolated PostgreSQL schemas during tests."""

from __future__ import annotations

import uuid
from typing import Final
from urllib.parse import ParseResult, quote, urlparse, urlunparse

import psycopg

__all__ = ["create_isolated_schema_uri", "isolated_test_schema_uri"]

_PUBLIC_SCHEMA: Final[str] = "public"


def _parse_query(query: str) -> dict[str, str]:
    if not query:
        return {}
    pairs = (part.split("=", 1) for part in query.split("&") if part)
    return {key: value for key, value in pairs if key}


def create_isolated_schema_uri(base_uri: str, schema_name: str) -> str:
    """Return ``base_uri`` with *schema_name* added to ``search_path``."""
    parsed: ParseResult = urlparse(base_uri)
    query_params = _parse_query(parsed.query)
    query_params["options"] = quote(f"-csearch_path={schema_name},{_PUBLIC_SCHEMA}")
    query = "&".join(f"{key}={value}" for key, value in query_params.items())
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))


def isolated_test_schema_uri(base_uri: str) -> str:
    """Create a temporary schema and return a connection URI that targets it."""
    schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"

    with psycopg.connect(base_uri) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema_name}")
            cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            cursor.execute(f"SET search_path TO {schema_name}, {_PUBLIC_SCHEMA}")
        conn.commit()

    return create_isolated_schema_uri(base_uri, schema_name)

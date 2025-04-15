from typing import Any, Dict

from omegaconf import OmegaConf

def all_fields():
    return "SELECT * FROM eval_data"


def total_metric(metric_field: str, filters: Dict[str, Any]):
    where_clause = build_where_clause(filters)
    query = f"""
        SELECT
            policy_name,
            eval_name,
            AVG(CAST("{metric_field}" AS DOUBLE)) AS mean_{metric_field.replace(".", "_")},
            STDDEV(CAST("{metric_field}" AS DOUBLE)) AS std_{metric_field.replace(".", "_")}
        FROM eval_data
        {where_clause}
        GROUP BY policy_name, eval_name
        ORDER BY policy_name, eval_name;"""

    return query


def build_where_clause(filters: Dict[str, Any]) -> str:
    """Build WHERE clause from a filters dictionary."""
    if not filters:
        return ""
    conditions = []

    # Convert OmegaConf objects to plain Python types if necessary.
    if OmegaConf.is_config(filters):
        filters = OmegaConf.to_container(filters, resolve=True)

    for field, value in filters.items():
        # If field names contain dots, wrap them in quotes.
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if "." in field and not field.startswith('"'):
            field = f'"{field}"'
        if isinstance(value, (list, tuple)):
            formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
            conditions.append(f"{field} IN ({', '.join(formatted_values)})")
        elif isinstance(value, str):
            value = value.strip()
            if value.startswith((">", "<", "=", "!=", ">=", "<=", "IN", "BETWEEN", "IS")):
                conditions.append(f"{field} {value}")
            else:
                conditions.append(f"{field} = '{value}'")
        else:
            conditions.append(f"{field} = {value}")
    return f"WHERE {' AND '.join(conditions)}" if conditions else ""

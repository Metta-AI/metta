# Wildcard Widget - Quick Reference

## Overview

Wildcard widgets enable custom Vega-Lite visualizations in Datadog dashboards. Fully supported via API.

**Live Example**: https://app.datadoghq.com/dashboard/bew-kg3-w4f/system-health-rollup-wildcard

## API Structure

The key is the `specification` wrapper pattern (not documented in official API docs):

```json
{
  "definition": {
    "type": "wildcard",
    "title": "My Chart",
    "requests": [
      {
        "response_format": "scalar",  // or "timeseries"
        "queries": [
          {
            "query": "avg:my.metric{*}",
            "data_source": "metrics",
            "name": "query1",
            "aggregator": "last"
          }
        ]
      }
    ],
    "specification": {
      "type": "vega-lite",           // ← REQUIRED (undocumented)
      "contents": {                   // ← Wrapper (undocumented)
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "table1"},   // References first query result
        "mark": "bar",
        "encoding": {
          "x": {"field": "query1", "type": "quantitative"},
          "y": {"field": "query1", "type": "quantitative"}
        }
      }
    }
  }
}
```

## Key Points

1. **Specification wrapper**: Vega-Lite spec goes inside `specification.contents` with `type: "vega-lite"`
2. **Data references**: Queries referenced as `table1`, `table2`, etc. (not `queryResults`)
3. **Vega-Lite version**: v5.18.1 supported
4. **Response formats**: `scalar` for aggregated values, `timeseries` for time-series data

## Common Pitfalls

❌ **Wrong**: Vega-Lite spec directly under `specification`
```json
"specification": {
  "$schema": "...",  // ← Missing wrapper!
  "data": {...}
}
```

✅ **Correct**: Wrapped in `type` + `contents`
```json
"specification": {
  "type": "vega-lite",
  "contents": {
    "$schema": "...",
    "data": {...}
  }
}
```

## Discovery Method

The correct structure was discovered by:
1. Creating a wildcard widget manually in Datadog UI
2. Exporting the dashboard JSON via API
3. Examining the widget definition structure

## Resources

- **Generator Script**: `devops/datadog/scripts/generate_wildcard_fom_grid.py`
- **Deployed Example**: `devops/datadog/templates/system_health_rollup_wildcard.json`
- **Vega-Lite Docs**: https://vega.github.io/vega-lite/docs/
- **Datadog Wildcard Widget Docs**: https://docs.datadoghq.com/dashboards/widgets/wildcard/

## FoM Grid Implementation

The `generate_wildcard_fom_grid.py` script creates a 7×7 heatmap showing:
- 7 CI FoM metrics (rows)
- 7 days of historical data (columns) using `.timeshift()`
- Color-coded cells (red/yellow/green)
- Text overlays with exact values
- Interactive tooltips

**Architecture**:
- 49 metric queries (7 metrics × 7 days)
- Vega-Lite `fold` transform to reshape data
- Layered visualization (rect marks + text marks)
- Threshold-based color scale

Run: `uv run python scripts/generate_wildcard_fom_grid.py` to regenerate.

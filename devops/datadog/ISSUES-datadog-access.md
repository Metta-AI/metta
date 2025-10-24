# Datadog Dashboard Development: Closing the Feedback Loop

## Problem Statement

Currently, Claude Code can:
- Generate dashboard configurations (Jsonnet/JSON)
- Push dashboards to Datadog via API
- Validate JSON structure and syntax

But Claude Code **cannot**:
- Verify that metric queries actually return data
- Detect when dashboards deploy successfully but show no data
- Know if a deployment worked without human checking Datadog UI

**This creates a critical gap**: We can deploy dashboards but have no automated way to verify the queries work with real data.

## Why This Matters

Without data verification:
1. **Silent Failures**: Dashboards may deploy successfully but show no data
2. **Query Errors**: Metric queries might be syntactically valid but semantically incorrect (wrong metric name, missing tags, etc.)
3. **Iteration Blindness**: Can't verify fixes without manual human inspection of Datadog UI

## Current Workflow Gap

```
┌─────────────────┐
│  Claude Code    │
│  - Generates    │
│  - Validates    │
│  - Pushes       │
└────────┬────────┘
         │
         │ API Push
         ▼
┌─────────────────┐
│  Datadog API    │
└────────┬────────┘
         │
         │ ??? (Gap)
         ▼
┌─────────────────┐
│  Datadog UI     │
│  (Human views)  │
└─────────────────┘

PROBLEM: No automated feedback from UI back to Claude Code
```

## Proposed Solution: Query Data Validation

**Approach**: Use Datadog's Metrics Query API to verify queries return data.

**What it does**:
- Extract metric queries from dashboard JSON
- Query each metric via API to check for data
- Report which queries work and which don't

**What it doesn't do**:
- Visual verification (layout, colors, formatting)
- Screenshot capture
- Check widget types or advanced features

**Why this is sufficient**:
- 90% of issues are "query returns no data"
- Visual issues are rare and easily caught by human spot-check
- Fast (<10 seconds) and simple to implement
- No browser automation complexity

## Implementation Plan

### Single Script: `verify_dashboard.py`

**Usage**:
```bash
# Verify a dashboard JSON file
python devops/datadog/scripts/verify_dashboard.py github_cicd.json

# Verify with custom time range
python devops/datadog/scripts/verify_dashboard.py github_cicd.json --hours=24
```

**What it does**:
1. Parse dashboard JSON
2. Extract all metric queries from widgets
3. For each query, call `/api/v1/query` API to check for data
4. Print simple report

**Output Example**:
```
Verifying dashboard: GitHub CI/CD
Found 8 widgets with queries

✓ avg:github.ci.duration{*} - 145 points, last: 2m ago
✓ sum:github.ci.failures{*} - 89 points, last: 5m ago
✗ avg:github.pr.review_time{*} - NO DATA
✓ count:github.commits{*} - 234 points, last: 1m ago
...

Summary: 7/8 queries returning data
Issue: github.pr.review_time metric not found or no recent data
```

**Implementation Sketch**:
```python
#!/usr/bin/env python3
"""Verify that dashboard queries return data."""

import json
import sys
import time
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi

def extract_queries(dashboard_json):
    """Extract all metric queries from dashboard."""
    queries = []
    for widget in dashboard_json.get('widgets', []):
        definition = widget.get('definition', {})
        for request in definition.get('requests', []):
            if 'q' in request:
                queries.append(request['q'])
    return queries

def verify_query(api, query, hours_back=1):
    """Check if query returns data."""
    now = int(time.time())
    start = now - (hours_back * 3600)

    try:
        response = api.query_metrics(_from=start, to=now, query=query)
        if not response.series:
            return {"status": "no_data", "points": 0}

        points = sum(len(s.pointlist) for s in response.series)
        last_ts = max(p[0] for s in response.series for p in s.pointlist) / 1000
        age = now - last_ts

        return {
            "status": "ok",
            "points": points,
            "age_seconds": age
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    dashboard_file = sys.argv[1]
    hours_back = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    with open(dashboard_file) as f:
        dashboard = json.load(f)

    queries = extract_queries(dashboard)
    print(f"Found {len(queries)} queries to verify\n")

    config = Configuration()
    with ApiClient(config) as api_client:
        api = MetricsApi(api_client)

        results = []
        for query in queries:
            result = verify_query(api, query, hours_back)
            results.append((query, result))

            if result['status'] == 'ok':
                age_min = int(result['age_seconds'] / 60)
                print(f"✓ {query[:60]} - {result['points']} points, last: {age_min}m ago")
            else:
                print(f"✗ {query[:60]} - {result['status'].upper()}")

        ok_count = sum(1 for _, r in results if r['status'] == 'ok')
        print(f"\nSummary: {ok_count}/{len(queries)} queries returning data")
```

## Research Findings

### Datadog Metrics Query API

**Endpoint**: `GET /api/v1/query`

**Status**: ✅ AVAILABLE - This is what we'll use

**Parameters**:
- `from`: UNIX epoch timestamp for start
- `to`: UNIX epoch timestamp for end
- `query`: Metric query string (e.g., `avg:system.cpu.user{host:web-01}`)

**Authentication**:
- Requires `DD-API-KEY` environment variable
- Requires `DD-APPLICATION-KEY` environment variable

**Response**: JSON with `series` field containing:
- Metric names
- Points (timestamp-value pairs)
- Associated tags

**Use Case**: Validate that queries return data after dashboard deployment

### What We Won't Implement

**Screenshots/Visual Verification**:
- Datadog has snapshot API for individual widgets, but it's complex (async, polling required)
- No full dashboard screenshot capability
- Not worth the effort - visual issues are rare and easily spotted by humans
- Focus on the 90% case: does the query return data?

### Questions to Answer During Implementation

- [ ] What are the rate limits for query API?
- [ ] How to handle queries that are slow to respond?
- [ ] Should we verify in parallel or sequential?
- [ ] What's a reasonable timeout per query?

## Success Criteria

A successful solution should:
1. **Fast feedback**: < 10 seconds to verify all queries
2. **Simple**: Single script, no complex dependencies
3. **Informative**: Clear output showing which queries work/don't work
4. **Integrated**: Easy to run from command line or Claude Code

## Implementation Steps

1. **Install dependency**:
   ```bash
   cd devops/datadog
   uv add datadog-api-client
   ```

2. **Create script**:
   ```bash
   touch devops/datadog/scripts/verify_dashboard.py
   chmod +x devops/datadog/scripts/verify_dashboard.py
   ```

3. **Implement basic functionality**:
   - Parse dashboard JSON
   - Extract queries
   - Call metrics API for each query
   - Print results

4. **Test with existing dashboards**:
   ```bash
   python devops/datadog/scripts/verify_dashboard.py devops/datadog/dashboards/templates/github_cicd.json
   ```

5. **Iterate based on real usage**:
   - Adjust time range if needed
   - Handle edge cases
   - Improve error messages

**Estimated effort**: 2-3 hours total

## Workflow Integration

**Current workflow**:
```bash
# Generate dashboard from Jsonnet
jsonnet dashboard.jsonnet > dashboard.json

# Push to Datadog
python devops/datadog/scripts/push_dashboard.py dashboard.json
```

**Enhanced workflow**:
```bash
# Generate dashboard from Jsonnet
jsonnet dashboard.jsonnet > dashboard.json

# Push to Datadog
python devops/datadog/scripts/push_dashboard.py dashboard.json

# Verify queries return data
python devops/datadog/scripts/verify_dashboard.py dashboard.json
```

**Claude Code usage**:
When Claude Code modifies a dashboard, it can automatically run the verification script after pushing to give immediate feedback about whether the queries work.

## Related Documentation

- Datadog API docs: https://docs.datadoghq.com/api/latest/
- Datadog dashboard JSON schema: https://docs.datadoghq.com/api/latest/dashboards/
- Datadog Snapshots API: https://docs.datadoghq.com/api/latest/snapshots/
- Datadog Metrics Query: https://docs.datadoghq.com/api/latest/metrics/
- Embeddable graphs: https://docs.datadoghq.com/dashboards/sharing/
- datadog-api-client (Python): https://github.com/DataDog/datadog-api-client-python

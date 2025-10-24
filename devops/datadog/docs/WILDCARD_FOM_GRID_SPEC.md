# Wildcard Widget FoM Grid Specification

## Overview

This document describes how to create a 7×7 FoM grid using Datadog's **Wildcard widget** with Vega-Lite, eliminating the need for external storage (S3) while providing full visual control.

## Solution Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Datadog Wildcard Widget                                 │
│                                                          │
│  ┌───────────────────┐     ┌──────────────────────┐    │
│  │ Datadog Queries   │────▶│ Vega-Lite Heatmap    │    │
│  │                   │     │                       │    │
│  │ - FoM metrics     │     │ - 7×7 grid           │    │
│  │ - Timeshift (-6d  │     │ - Color encoding     │    │
│  │   to today)       │     │ - Text overlays      │    │
│  └───────────────────┘     └──────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Benefits Over Other Approaches

| Approach | Storage | Control | Interactive | Complexity |
|----------|---------|---------|-------------|------------|
| **Widget Grid** (current) | ✅ Datadog | ❌ Limited | ✅ Yes | ✅ Low |
| **Image Collector** | ❌ S3 | ✅ Full | ❌ No | ❌ High |
| **Wildcard Widget** | ✅ Datadog | ✅ Full | ✅ Yes | ✅ Medium |

## Data Requirements

### Metrics to Query

For each of 7 metrics, we need to query 7 time points:

```
health.ci.tests_passing.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.failing_workflows.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.hotfix_count.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.revert_count.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.duration_p90.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.stale_prs.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
health.ci.pr_cycle_time.fom (today, -1d, -2d, -3d, -4d, -5d, -6d)
```

Total: **49 query-day combinations**

### Data Format Expected by Vega-Lite

```json
[
  {"metric": "Tests Passing", "day": "Today", "value": 0.95},
  {"metric": "Tests Passing", "day": "-1d", "value": 0.92},
  {"metric": "Tests Passing", "day": "-2d", "value": 0.88},
  ...
  {"metric": "PR Cycle Time", "day": "-6d", "value": 0.45}
]
```

## Challenge: Datadog Query Limitations

### Problem

Datadog queries return data in this format:

```json
{
  "response_format": "scalar",
  "queries": [
    {
      "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-1d)",
      "data_source": "metrics",
      "name": "tests_passing_1d",
      "aggregator": "last"
    }
  ]
}
```

Result: `{"tests_passing_1d": 0.92}`

**Issue**: This returns 49 separate values, but Vega-Lite needs structured data with metric name, day, and value in each row.

### Solution Options

#### Option A: Transform in Vega-Lite (Recommended)

Use Vega-Lite's **transform** capability to reshape the data:

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "data": {
    "values": [
      {"tests_passing_today": 0.95, "tests_passing_1d": 0.92, ...}
    ]
  },
  "transform": [
    {
      "fold": [
        "tests_passing_today", "tests_passing_1d", "tests_passing_2d",
        "failing_workflows_today", "failing_workflows_1d", ...
      ],
      "as": ["metric_day", "value"]
    },
    {
      "calculate": "split(datum.metric_day, '_')[0] + ' ' + split(datum.metric_day, '_')[1]",
      "as": "metric"
    },
    {
      "calculate": "split(datum.metric_day, '_')[2]",
      "as": "day"
    }
  ],
  "mark": "rect",
  "encoding": {
    "y": {"field": "metric", "type": "nominal", "title": "Metric"},
    "x": {"field": "day", "type": "ordinal", "title": null,
          "sort": ["6d", "5d", "4d", "3d", "2d", "1d", "today"]},
    "color": {
      "field": "value",
      "type": "quantitative",
      "scale": {
        "domain": [0, 0.3, 0.7, 1.0],
        "range": ["#dc3545", "#ffc107", "#ffc107", "#28a745"],
        "type": "threshold"
      },
      "legend": {"title": "FoM"}
    }
  }
}
```

#### Option B: Multiple Queries with Explicit Mapping

Create 49 separate queries and manually map to grid positions (verbose but explicit).

## Proof of Concept Specification

### Complete Wildcard Widget Configuration

```json
{
  "definition": {
    "title": "System Health FoM Grid (Wildcard)",
    "type": "wildcard",
    "requests": [
      {
        "response_format": "scalar",
        "queries": [
          {
            "query": "avg:health.ci.tests_passing.fom{*}",
            "data_source": "metrics",
            "name": "tests_passing_today",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-1d)",
            "data_source": "metrics",
            "name": "tests_passing_1d",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-2d)",
            "data_source": "metrics",
            "name": "tests_passing_2d",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-3d)",
            "data_source": "metrics",
            "name": "tests_passing_3d",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-4d)",
            "data_source": "metrics",
            "name": "tests_passing_4d",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-5d)",
            "data_source": "metrics",
            "name": "tests_passing_5d",
            "aggregator": "last"
          },
          {
            "query": "avg:health.ci.tests_passing.fom{*}.timeshift(-6d)",
            "data_source": "metrics",
            "name": "tests_passing_6d",
            "aggregator": "last"
          }
          // Repeat for other 6 metrics (42 more queries)
        ]
      }
    ],
    "custom_viz": {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "data": {"name": "queryResults"},
      "transform": [
        {
          "fold": [
            "tests_passing_today", "tests_passing_1d", "tests_passing_2d",
            "tests_passing_3d", "tests_passing_4d", "tests_passing_5d", "tests_passing_6d",
            "failing_workflows_today", "failing_workflows_1d", "failing_workflows_2d",
            "failing_workflows_3d", "failing_workflows_4d", "failing_workflows_5d", "failing_workflows_6d",
            "hotfix_count_today", "hotfix_count_1d", "hotfix_count_2d",
            "hotfix_count_3d", "hotfix_count_4d", "hotfix_count_5d", "hotfix_count_6d",
            "revert_count_today", "revert_count_1d", "revert_count_2d",
            "revert_count_3d", "revert_count_4d", "revert_count_5d", "revert_count_6d",
            "duration_p90_today", "duration_p90_1d", "duration_p90_2d",
            "duration_p90_3d", "duration_p90_4d", "duration_p90_5d", "duration_p90_6d",
            "stale_prs_today", "stale_prs_1d", "stale_prs_2d",
            "stale_prs_3d", "stale_prs_4d", "stale_prs_5d", "stale_prs_6d",
            "pr_cycle_time_today", "pr_cycle_time_1d", "pr_cycle_time_2d",
            "pr_cycle_time_3d", "pr_cycle_time_4d", "pr_cycle_time_5d", "pr_cycle_time_6d"
          ],
          "as": ["metric_day", "value"]
        },
        {
          "calculate": "replace(split(datum.metric_day, '_today')[0], '_', ' ')",
          "as": "metric_name"
        },
        {
          "calculate": "indexof(datum.metric_day, 'today') > 0 ? 'Today' : (indexof(datum.metric_day, '1d') > 0 ? '-1d' : (indexof(datum.metric_day, '2d') > 0 ? '-2d' : (indexof(datum.metric_day, '3d') > 0 ? '-3d' : (indexof(datum.metric_day, '4d') > 0 ? '-4d' : (indexof(datum.metric_day, '5d') > 0 ? '-5d' : '-6d')))))",
          "as": "day"
        }
      ],
      "layer": [
        {
          "mark": "rect",
          "encoding": {
            "y": {
              "field": "metric_name",
              "type": "nominal",
              "title": null,
              "sort": [
                "tests passing",
                "failing workflows",
                "hotfix count",
                "revert count",
                "duration p90",
                "stale prs",
                "pr cycle time"
              ]
            },
            "x": {
              "field": "day",
              "type": "ordinal",
              "title": null,
              "sort": ["-6d", "-5d", "-4d", "-3d", "-2d", "-1d", "Today"]
            },
            "color": {
              "field": "value",
              "type": "quantitative",
              "scale": {
                "domain": [0, 0.3, 0.7, 1.0],
                "range": ["#dc3545", "#ffc107", "#ffc107", "#28a745"],
                "type": "threshold"
              },
              "legend": {"title": "FoM Score"}
            }
          }
        },
        {
          "mark": {"type": "text", "fontSize": 14, "fontWeight": "bold"},
          "encoding": {
            "y": {
              "field": "metric_name",
              "type": "nominal",
              "sort": [
                "tests passing",
                "failing workflows",
                "hotfix count",
                "revert count",
                "duration p90",
                "stale prs",
                "pr cycle time"
              ]
            },
            "x": {
              "field": "day",
              "type": "ordinal",
              "sort": ["-6d", "-5d", "-4d", "-3d", "-2d", "-1d", "Today"]
            },
            "text": {"field": "value", "type": "quantitative", "format": ".2f"},
            "color": {
              "condition": {"test": "datum.value < 0.5", "value": "white"},
              "value": "black"
            }
          }
        }
      ],
      "config": {
        "axis": {"grid": true, "tickBand": "extent"}
      }
    }
  }
}
```

## Implementation Steps

### Phase 1: Proof of Concept (2 hours)

1. **Create single-metric test** (15 min)
   - Query "Tests Passing" for 7 days
   - Create 1×7 heatmap
   - Verify Vega-Lite transform works

2. **Expand to full grid** (30 min)
   - Add all 7 metrics
   - Create 49 queries
   - Verify data transformation

3. **Add visual polish** (45 min)
   - Text overlays with values
   - Color scheme matching current grid
   - Responsive sizing
   - Legend and labels

4. **Deploy and test** (30 min)
   - Push to Datadog
   - Verify metrics display correctly
   - Compare with current widget grid

### Phase 2: Refinement (1 hour)

1. **Optimize query performance**
   - Combine queries if possible
   - Test update latency

2. **Add interactivity**
   - Tooltips with metric details
   - Click-through to detailed views

3. **Documentation**
   - Update DASHBOARD_WORKPLAN.md
   - Create maintenance guide

## Open Questions

1. **Wildcard widget availability**: Is it available in our Datadog plan?
2. **Query limits**: Can we make 49 metric queries in one widget?
3. **Transform complexity**: Will the nested calculate expressions work?
4. **Update frequency**: How often does the Wildcard widget refresh?

## Comparison: Current Grid vs Wildcard

| Aspect | Widget Grid | Wildcard Widget |
|--------|-------------|-----------------|
| **Storage** | ✅ Datadog metrics | ✅ Datadog metrics |
| **Setup complexity** | Low (65 widgets) | Medium (1 widget, complex spec) |
| **Visual flexibility** | Limited | High |
| **Text labels** | ❌ No | ✅ Yes |
| **Grid alignment** | Manual positioning | Automatic |
| **Interactivity** | Basic (hover) | Advanced (Vega events) |
| **Maintenance** | Edit generator script | Edit Vega spec |
| **Dashboard width** | 10/12 columns | Flexible |

## Recommendation

**Try the Wildcard widget POC** for these reasons:

1. ✅ **No S3 dependency** - Datadog manages all data
2. ✅ **Full visual control** - Vega-Lite is very flexible
3. ✅ **Text overlays** - Show exact FoM values
4. ✅ **Better alignment** - Automatic grid layout
5. ✅ **One widget** - Easier to maintain than 65 widgets
6. ⚠️ **Medium complexity** - But only needs to be built once

If it works well, we get the best of both worlds: Datadog-native storage + custom visualization.

If it doesn't work (e.g., widget not available, too slow, transform issues), we keep the current widget grid approach which already works.

**Estimated time: 2-3 hours total for POC**

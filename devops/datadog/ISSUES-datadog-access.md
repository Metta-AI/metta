# Datadog Dashboard Development: Closing the Feedback Loop

## Problem Statement

Currently, Claude Code can:
- Generate dashboard configurations (Jsonnet/JSON)
- Push dashboards to Datadog via API
- Validate JSON structure and syntax

But Claude Code **cannot**:
- View the rendered dashboard in Datadog UI
- Verify that collected metrics are actually displayed correctly
- Detect visualization issues (wrong queries, missing data, incorrect formatting)
- Iterate on dashboard improvements based on actual data rendering

**This creates a critical gap**: We can deploy dashboards but have no automated way to verify they work correctly with real data.

## Why This Matters

Without visual verification:
1. **Silent Failures**: Dashboards may deploy successfully but show no data or wrong data
2. **Query Errors**: Metric queries might be syntactically valid but semantically incorrect
3. **Layout Issues**: Widgets might overlap, be incorrectly sized, or poorly organized
4. **Data Gaps**: Missing tags, wrong aggregations, or incorrect time ranges
5. **Iteration Blindness**: Can't verify fixes without manual human inspection

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

## Potential Solutions

### 1. Screenshot/Snapshot API

**Approach**: Capture rendered dashboard as image for visual inspection.

**Pros**:
- Direct visual verification
- Catch layout and rendering issues
- Can use vision models (Claude) to analyze output

**Cons**:
- Datadog may not have public screenshot API
- Screenshots don't show interactivity
- May require browser automation (Playwright/Puppeteer)

**Research Needed**:
- Does Datadog have a dashboard export/screenshot API?
- Can we use their embeddable dashboard feature?

### 2. Query Results API

**Approach**: Fetch the actual metric query results that would populate widgets.

**Pros**:
- Verify data availability without rendering
- Can check if queries return expected number of results
- Programmatically validate data shape

**Cons**:
- Doesn't verify visual rendering
- May miss layout/formatting issues
- Need to know expected query results

**Implementation**:
```python
# Fetch query results for each widget
for widget in dashboard['widgets']:
    query = widget['definition']['requests'][0]['q']
    results = dd_api.query_metrics(query, start_time, end_time)
    assert len(results) > 0, f"No data for query: {query}"
```

**Research Needed**:
- What's the Datadog metrics query API endpoint?
- How do we handle different widget types (timeseries, heatmap, etc.)?

### 3. Dashboard JSON Validation API

**Approach**: Use Datadog's validation endpoint (if exists) to pre-validate before push.

**Pros**:
- Catch errors before deployment
- Fast feedback loop
- No visual inspection needed for basic validation

**Cons**:
- Only catches schema errors, not data issues
- Doesn't verify metrics exist or have data

**Research Needed**:
- Does Datadog have a validation-only API endpoint?
- Can we get validation warnings/errors without creating dashboard?

### 4. Browser Automation with Screenshot

**Approach**: Use Playwright/Puppeteer to login, navigate to dashboard, capture screenshot.

**Pros**:
- Full rendering verification
- Can interact with dashboard (hover, click, etc.)
- Can capture multiple views/time ranges

**Cons**:
- Requires browser automation setup
- Slower than API-only approaches
- Need to handle Datadog authentication
- May violate Datadog ToS if automated at scale

**Implementation Sketch**:
```python
from playwright.sync_api import sync_playwright

def capture_dashboard(dashboard_id: str) -> bytes:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Authenticate (how?)
        page.goto(f"https://app.datadoghq.com/dashboard/{dashboard_id}")

        # Wait for widgets to load
        page.wait_for_selector('[data-testid="dashboard-widget"]')

        # Capture screenshot
        screenshot = page.screenshot()
        browser.close()
        return screenshot
```

**Challenges**:
- How to authenticate? (API key? OAuth? Session cookies?)
- How to wait for all widgets to load?
- How long to wait for time-series data?

### 5. Datadog Public Dashboard Sharing

**Approach**: Create public-shareable dashboards, fetch via public URL.

**Pros**:
- No authentication needed for viewing
- Can use simple HTTP fetch + screenshot
- Embeddable in CI/CD reports

**Cons**:
- Not all dashboards should be public
- Security concerns with sensitive metrics
- May not work for all widget types

### 6. Synthetic Monitoring Integration

**Approach**: Use Datadog's Synthetic Monitoring to test dashboard loading.

**Pros**:
- Native Datadog feature
- Can alert on dashboard load failures
- Tracks historical dashboard health

**Cons**:
- Doesn't help Claude Code iterate during development
- Requires separate setup per dashboard
- May incur additional Datadog costs

## Recommended Hybrid Approach

Combine multiple techniques for comprehensive validation:

### Phase 1: Deployment Validation (Fast, API-only)
1. **Pre-push validation**: Validate JSON schema locally
2. **Query testing**: Test each metric query returns data
3. **Push dashboard**: Deploy via API
4. **Verify push**: Confirm dashboard ID returned

### Phase 2: Visual Verification (Slower, optional)
5. **Screenshot capture**: Use browser automation or screenshot API
6. **AI analysis**: Use Claude vision to analyze screenshot for obvious issues
7. **Report results**: Generate markdown report with findings

### Example Workflow
```bash
# 1. Push dashboard
python devops/datadog/scripts/push_dashboard.py github_cicd.json

# 2. Validate deployment
python devops/datadog/scripts/verify_dashboard.py <dashboard_id> \
  --check-queries \
  --check-data \
  --screenshot \
  --report=verification_report.md
```

## Questions to Research

### Datadog API Capabilities
- [ ] Does Datadog have a screenshot/export API for dashboards?
- [ ] What's the best way to test metric queries programmatically?
- [ ] Can we embed dashboards in external tools without authentication?
- [ ] Is there a validation-only API endpoint?

### Authentication & Access
- [ ] How to authenticate browser automation to Datadog?
- [ ] Can we create read-only API keys for verification?
- [ ] What are the rate limits for dashboard API calls?

### Data Verification
- [ ] How to determine if a metric query "looks right"?
- [ ] What's a reasonable data freshness expectation?
- [ ] How to handle dashboards with historical data vs. real-time data?

### Claude Code Integration
- [ ] Can Claude Code analyze screenshots directly?
- [ ] How to present verification results in CLI?
- [ ] Should verification be automatic or on-demand?

## Success Criteria

A successful solution should:
1. **Fast feedback**: < 30 seconds to verify dashboard is working
2. **Comprehensive**: Catch both data and visual issues
3. **Automated**: Minimal manual intervention required
4. **Informative**: Clear error messages when issues found
5. **Integrated**: Works seamlessly in Claude Code workflow

## Next Steps

1. **Research Datadog API documentation** for screenshot/export capabilities
2. **Prototype query validation** script to test metric data availability
3. **Experiment with browser automation** for screenshot capture
4. **Design verification report format** that Claude Code can interpret
5. **Build minimal verification tool** that integrates into existing workflow

## Related Documentation

- Datadog API docs: https://docs.datadoghq.com/api/latest/
- Datadog dashboard JSON schema: https://docs.datadoghq.com/api/latest/dashboards/
- Embeddable graphs: https://docs.datadoghq.com/dashboards/sharing/

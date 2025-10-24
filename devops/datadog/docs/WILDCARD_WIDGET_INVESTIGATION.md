# Wildcard Widget API Investigation

## Summary

Attempted comprehensive investigation of Datadog Wildcard widget API support to enable programmatic creation of custom Vega-Lite visualizations.

**Conclusion**: Wildcard widget appears to be **UI-only** with limited or no API support as of 2025-10-23.

## Investigation Steps

### 1. Documentation Search

✅ **Found**: Extensive documentation about Wildcard widgets exists:
- Main widget page: https://docs.datadoghq.com/dashboards/widgets/wildcard/
- Getting started guide: https://docs.datadoghq.com/dashboards/guide/getting_started_with_wildcard_widget/
- Examples gallery: https://docs.datadoghq.com/dashboards/guide/wildcard_examples/
- Vega-Lite usage: https://docs.datadoghq.com/dashboards/guide/using_vega_lite_in_wildcard_widgets/

✅ **Confirms**:
- Wildcard widget uses Vega-Lite v5.18.1
- Supports custom visualizations beyond native widgets
- Data sources referenced as `table1`, `table2`, etc.
- Uses standard Vega-Lite grammar with Datadog extensions

❌ **Missing**: No JSON/API examples in any documentation

### 2. API Client Libraries

**Python Client** (`datadog-api-client-python`):
- ❌ No `WildcardWidgetDefinition` class found
- ❌ No examples of creating wildcard widgets programmatically
- ✅ Has definitions for all other widget types (timeseries, query_value, heatmap, etc.)

**TypeScript/JavaScript Client**:
- Same pattern - no wildcard widget classes found

**Terraform Provider** (`terraform-provider-datadog`):
- ❌ No wildcard widget examples in `examples/resources/datadog_dashboard/`
- ✅ Documentation mentions using `datadog_dashboard_json` for unsupported widgets
- ⚠️ Still requires knowing the correct JSON structure

### 3. GitHub Code Search

Searched for:
- `"type": "wildcard"` in dashboard JSON files
- `wildcard_widget_definition.py` in Python client
- Exported dashboard JSONs with wildcard widgets
- Terraform HCL examples

**Result**: Zero examples found

### 4. Direct API Testing

Attempted to create wildcard widget via Datadog API:

```python
widget = {
    "definition": {
        "type": "wildcard",
        "title": "Test",
        "requests": [{"response_format": "scalar", "queries": [...]}],
        "specification": {  # Vega-Lite spec
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"name": "table1"},
            "mark": "bar",
            "encoding": {...}
        }
    }
}
```

**Error**:
```
Status Code: 400
Error: "Invalid widget definition at position 0 of type wildcard.
        Error: 'type' is a required property."
```

Variations tried:
- Changed `specification` → `custom_viz`
- Changed `custom_viz` → `custom_definition`
- Used minimal test cases
- Followed documented Vega-Lite structure

**All failed with same validation error**

## Evidence Analysis

| Evidence | Conclusion |
|----------|------------|
| Extensive UI documentation | Widget exists and is supported in UI |
| No API client classes | Not supported in official client libraries |
| No API examples anywhere | Not publicly documented for API use |
| No GitHub examples | No one has successfully exported one |
| API validation errors | Widget definition structure differs from docs |
| Terraform uses `dashboard_json` fallback | Even Terraform can't use type-safe definitions |

## Possible Explanations

1. **UI-Only Feature**: Wildcard widget may be intentionally UI-only
   - Requires visual editor for Vega-Lite development
   - Too complex for pure JSON/API workflow

2. **Beta/New Feature**: May be too new for API support
   - Released in UI first
   - API support coming later
   - Client libraries lag behind

3. **Undocumented API Structure**: Correct structure exists but isn't documented
   - Internal Datadog teams may know the format
   - Community hasn't reverse-engineered it yet

4. **Enterprise/Plan-Gated**: May require specific Datadog plan tier
   - Not available via API in our plan
   - Different validation rules per plan

## Recommendation

### Approach 1: Manual Creation + Export (Recommended)

1. Create wildcard widget manually in Datadog UI:
   - Go to dashboard
   - Add widget → Wildcard
   - Build Vega-Lite visualization in UI editor

2. Export dashboard JSON:
   ```bash
   source ./load_env.sh
   ./scripts/export_dashboard.py <dashboard-id> > exported.json
   ```

3. Extract wildcard widget structure from JSON

4. Use extracted structure as template for programmatic generation

**Pros**:
- Will reveal actual API structure
- Confirms if widget can be exported
- Provides working example to build from

**Cons**:
- Defeats purpose of programmatic generation
- May still fail on re-import if structure is UI-specific

### Approach 2: Contact Datadog Support

Questions to ask:
1. Is Wildcard widget supported via Dashboard API?
2. What is the correct JSON structure for API creation?
3. Are there any plan/tier restrictions?
4. Is API support planned for future releases?

### Approach 3: Stick with Widget Grid (Current)

**Status**: Already deployed and working
**URL**: https://app.datadoghq.com/dashboard/2mx-kfj-8pi/system-health-rollup

**Advantages**:
- ✅ Proven to work via API
- ✅ Datadog manages all data (no S3)
- ✅ Fully programmatic generation
- ✅ Conditional color formatting
- ✅ Interactive hover

**Trade-offs**:
- ⚠️ 65 widgets vs 1 (maintenance overhead manageable)
- ⚠️ No text labels (can add note widgets if needed)
- ⚠️ Manual alignment (already done)

## Next Steps

**Option A** (Recommended): Accept widget grid approach
- It meets all requirements (Datadog-managed data, no S3)
- Works reliably via API
- Minor maintenance overhead is acceptable
- Can revisit wildcard widget when API support is confirmed

**Option B**: Manual creation experiment
- Create one wildcard widget in UI
- Export dashboard to see JSON structure
- Attempt to re-import
- Document findings

**Option C**: Contact Datadog
- Open support ticket
- Ask about API support status
- Wait for official response

## Files Created During Investigation

- `devops/datadog/docs/WILDCARD_FOM_GRID_SPEC.md` - Architecture plan (275 lines)
- `devops/datadog/scripts/generate_wildcard_fom_grid.py` - Generator script (273 lines)
- `devops/datadog/templates/system_health_rollup_wildcard.json` - Generated (not deployable)
- `devops/datadog/docs/DASHBOARD_WORKPLAN.md` - Updated with blocker details
- `devops/datadog/docs/WILDCARD_WIDGET_INVESTIGATION.md` - This document

## References

- Datadog Wildcard Widget Docs: https://docs.datadoghq.com/dashboards/widgets/wildcard/
- Vega-Lite v5 Spec: https://vega.github.io/schema/vega-lite/v5.json
- Datadog API Docs: https://docs.datadoghq.com/api/latest/dashboards/
- Current Working Dashboard: https://app.datadoghq.com/dashboard/2mx-kfj-8pi/system-health-rollup

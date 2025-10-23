# Modular Dashboard Workflow

## Philosophy: Mix and Match Components

Instead of managing full dashboards as monolithic JSON blobs, we want to:

1. **Break dashboards into reusable widget components**
2. **Mix and match** widgets to create custom views
3. **Share widgets** across different dashboards
4. **Version control** individual components, not just full dashboards

## Proposed Workflow

### Current State (What We Have)

```
templates/
├── softmax_system_health.json       # Full dashboard (3 widgets)
├── policy_evaluator.json            # Full dashboard (3 widgets)
└── softmax_pulse.json               # Full dashboard (0 widgets)
```

**Problem:** To reuse a widget, you have to copy/paste JSON between dashboard files.

### Improved State (What We Want)

```
components/                          # Reusable widget definitions
├── ci/
│   ├── tests_passing.json          # Single widget
│   ├── reverts_count.json          # Single widget
│   └── hotfix_count.json           # Single widget
├── apm/
│   ├── orchestrator_latency.json   # Single widget
│   ├── worker_latency.json         # Single widget
│   └── worker_hits.json            # Single widget
└── README.md                        # Component catalog

dashboards/                          # Dashboard compositions
├── softmax_health.yaml              # References components
├── policy_eval.yaml                 # References components
└── custom_view.yaml                 # Mix and match!
```

## Example: Component-Based Dashboard

### Widget Component (Single Reusable Unit)

`components/ci/tests_passing.json`:
```json
{
  "widget_type": "timeseries",
  "title": "Tests are passing on main",
  "query": "avg:ci.tests_passing_on_main{source:softmax-system-health}",
  "markers": [
    {
      "label": "Unit-test jobs should be passing on main",
      "value": "y = 1",
      "display_type": "info"
    }
  ],
  "layout": {
    "width": 6,
    "height": 3
  }
}
```

### Dashboard Composition (References Components)

`dashboards/softmax_health.yaml`:
```yaml
title: Softmax System Health
description: CI/CD health metrics
layout_type: ordered

widgets:
  - component: ci/tests_passing
    position: {x: 0, y: 0}

  - component: ci/reverts_count
    position: {x: 6, y: 0}

  - component: ci/hotfix_count
    position: {x: 6, y: 3}
```

### Custom Mix-and-Match Dashboard

`dashboards/my_custom_view.yaml`:
```yaml
title: My Custom Monitoring View
description: Mix of CI and APM metrics
layout_type: ordered

widgets:
  - component: ci/tests_passing
  - component: apm/orchestrator_latency
  - component: ci/hotfix_count
```

## Benefits

✅ **Reusability** - Write widget once, use in many dashboards
✅ **Consistency** - Same widget looks identical across dashboards
✅ **Discoverability** - Browse component catalog to find widgets
✅ **Version Control** - See when individual widgets change
✅ **Easier Updates** - Update widget in one place, affects all dashboards using it
✅ **Mix and Match** - Create custom views easily

## Implementation Plan

### Phase 1: Extract Components (Manual)

```bash
# 1. Pull existing dashboards
make pull

# 2. Manually extract widgets to components/
mkdir -p components/ci components/apm

# 3. Create component files (one widget each)
# ... manual extraction ...

# 4. Create dashboard compositions (YAML)
# ... reference components ...
```

### Phase 2: Build Assembler

Create a script that assembles dashboards from components:

```bash
# Build dashboard from composition
./assemble_dashboard.py dashboards/softmax_health.yaml > templates/softmax_health.json

# Push to Datadog
make push
```

### Phase 3: Automated Workflow

```bash
# 1. Edit components or compositions
vim components/ci/tests_passing.json
vim dashboards/softmax_health.yaml

# 2. Assemble all dashboards
make assemble

# 3. Push to Datadog
make push
```

## Workflow Comparison

### Current Workflow (Monolithic)

```bash
make pull                                    # Download full dashboards
vim templates/softmax_system_health.json     # Edit 200-line JSON
# (find widget buried in JSON, edit carefully)
make diff                                    # See changes
make push                                    # Upload
```

**Problem:** Hard to find/edit specific widgets, no reuse.

### Proposed Workflow (Modular)

```bash
# Option 1: Edit component (affects all dashboards using it)
vim components/ci/tests_passing.json         # Edit 20-line JSON
make assemble                                # Rebuild dashboards
make push                                    # Upload

# Option 2: Create new dashboard composition
vim dashboards/my_view.yaml                  # List widgets to include
make assemble                                # Build dashboard JSON
make push                                    # Upload
```

**Benefits:** Clear, reusable, easy to mix-and-match.

## Questions to Consider

### 1. Component Granularity

**Q:** Should each widget be a component, or should we have larger building blocks?

**Options:**
- **Fine-grained** - One widget = one component (most flexible)
- **Coarse-grained** - Related widgets grouped (easier to manage)
- **Both** - Support both levels

**Recommendation:** Start with fine-grained (one widget = one component) for maximum flexibility.

### 2. Composition Format

**Q:** What format for dashboard compositions?

**Options:**
- **YAML** - Human-readable, easy to edit
- **JSON** - Consistent with Datadog format
- **Python/Config** - Most flexible, but complex

**Recommendation:** YAML for readability, generates JSON for Datadog.

### 3. Discovery

**Q:** How do users find available components?

**Options:**
- **README catalog** - Manually maintained list
- **Auto-generated catalog** - Script scans components/
- **CLI tool** - `./list_components.py --category=ci`

**Recommendation:** Start with README, add CLI tool later.

## Next Steps

1. **Create `list_metrics.py`** ✅ (done!)
2. **Extract a few components manually** (prototype)
3. **Create `assemble_dashboard.py`** (script to build from components)
4. **Build dashboards** with `metta datadog dashboard build`
5. **Document component structure** (README in components/)
6. **Test workflow** with real dashboards

## Example Commands (Future)

```bash
# Discovery
metta datadog dashboard metrics              # Show available metrics
# (list-components not yet implemented)

# Component workflow
vim components/ci/my_widget.json
metta datadog dashboard build                # Rebuild dashboards from components
metta datadog dashboard diff                 # Review changes
metta datadog dashboard push                 # Upload to Datadog

# Dashboard workflow
vim dashboards/my_view.yaml    # Create new composition
metta datadog dashboard build                # Build JSON
metta datadog dashboard push                 # Upload to Datadog
```

## Trade-offs

### Pros
- ✅ Maximum reusability
- ✅ Easy to mix-and-match
- ✅ Clear component ownership
- ✅ Better version control diffs
- ✅ Discoverable via catalog

### Cons
- ❌ Additional complexity (assembler needed)
- ❌ More files to manage
- ❌ Learning curve for team
- ❌ Requires build step (`metta datadog dashboard build`)

## Decision Point

**Should we proceed with modular approach?**

If **YES**:
- Build assembler script
- Extract components from existing dashboards
- Update documentation

If **NO** (keep current approach):
- Stick with full dashboard JSON files
- Use comments/sections to organize
- Copy/paste widgets when needed

**What do you think?** The modular approach is more complex but gives you the "mix and match" capability you want.

---

Last updated: 2025-10-22

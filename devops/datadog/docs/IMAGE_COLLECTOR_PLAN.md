# Image-Based Derived Data Collector - Architecture Plan

> **⚠️ NOT IMPLEMENTED - REJECTED APPROACH**
>
> This approach was rejected because it requires external storage (S3/Imgur).
> **Alternative used**: Wildcard widget with Vega-Lite (see `WILDCARD_WIDGET.md`)
>
> This document is kept for reference only.

---

## Overview

A new collector type that generates visualizations (images) from Datadog metrics and displays them on dashboards. This enables pixel-perfect custom visualizations while leveraging existing metric infrastructure.

**Status**: ❌ Rejected (requires S3)
**Type**: Derived/Calculated data collector
**Output**: Images displayed via Datadog image widgets

---

## Motivation

### Current Limitations
- **Widget constraints**: Datadog's native widgets have layout limitations (12-column grid, limited customization)
- **Complex visualizations**: Hard to create custom heatmaps, annotated charts, or multi-dimensional views
- **Exact specifications**: Difficult to match exact visual requirements (like the ASCII grid spec)

### Image Collector Benefits
- ✅ **Pixel-perfect control**: Complete freedom with matplotlib/seaborn
- ✅ **Custom annotations**: Add trends, thresholds, sparklines, anything
- ✅ **Compact display**: Single image widget vs 50+ query_value widgets
- ✅ **Familiar tools**: Python data viz ecosystem (matplotlib, seaborn, plotly)
- ✅ **Exactly matches specs**: Can replicate the original ASCII table perfectly

### Tradeoffs
- ❌ **Static snapshots**: Updates only when collector runs (not real-time)
- ❌ **No interactivity**: Can't hover for details or drill down
- ❌ **External hosting**: Need S3/Imgur/CDN for image storage
- ❌ **Can't use Datadog time selector**: Image shows fixed time range

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Image Collector Pipeline                     │
└─────────────────────────────────────────────────────────────────┘

1. FETCH DATA                    2. GENERATE IMAGE
   ┌──────────────┐                 ┌──────────────┐
   │ Query Datadog│─────────────────▶│  Matplotlib  │
   │   Metrics    │  FoM values      │  Heatmap     │
   │              │  (7 days × 7     │  Generation  │
   │ - health.ci. │   metrics)       │              │
   │   *.fom      │                  │ - Seaborn    │
   └──────────────┘                 │ - Custom     │
                                    │   styling    │
                                    └──────┬───────┘
                                           │
3. UPLOAD IMAGE                            │ PNG file
   ┌──────────────┐                        │
   │  S3 / Imgur  │◀───────────────────────┘
   │  Upload      │
   │              │
   │ Returns URL: │
   │ https://...  │
   └──────┬───────┘
          │
4. DISPLAY ON DASHBOARD
   ┌──────────────┐
   │ Datadog      │
   │ Image Widget │◀─── References image URL
   │              │
   └──────────────┘
```

### Component Design

#### 1. Base Class: `ImageCollector`

```python
from abc import abstractmethod
from devops.datadog.common.base import BaseCollector
from devops.datadog.common.datadog_client import DatadogClient

class ImageCollector(BaseCollector):
    """Base class for collectors that generate visualization images.

    Unlike standard collectors that emit metrics, ImageCollectors:
    1. Fetch existing metrics from Datadog
    2. Generate visualizations (matplotlib/seaborn)
    3. Upload images to storage (S3/Imgur)
    4. Optionally emit metadata metrics (image URL, generation time)
    """

    def __init__(self, name: str, datadog_client: DatadogClient,
                 image_uploader: ImageUploader):
        super().__init__(name=name)
        self.dd_client = datadog_client
        self.uploader = image_uploader

    def collect_metrics(self) -> dict[str, Any]:
        """Main collection logic: fetch → generate → upload."""
        try:
            # 1. Fetch source data from Datadog
            self.logger.info("Fetching source data from Datadog...")
            data = self.fetch_source_data()

            # 2. Generate visualization
            self.logger.info("Generating visualization...")
            image_path = self.generate_image(data)

            # 3. Upload to storage
            self.logger.info("Uploading image...")
            image_url = self.uploader.upload(image_path)

            # 4. Return metadata metrics
            return {
                f"{self.name}.image_url": image_url,
                f"{self.name}.updated_at": time.time(),
                f"{self.name}.generation_success": 1.0,
            }

        except Exception as e:
            self.logger.error(f"Failed to generate image: {e}")
            return {
                f"{self.name}.generation_success": 0.0,
            }

    @abstractmethod
    def fetch_source_data(self) -> dict:
        """Fetch metrics from Datadog API.

        Returns:
            Dict of metric data needed for visualization
        """
        pass

    @abstractmethod
    def generate_image(self, data: dict) -> str:
        """Generate visualization and save to file.

        Args:
            data: Metric data from fetch_source_data()

        Returns:
            Path to generated image file
        """
        pass
```

#### 2. Image Uploaders

**Interface:**
```python
class ImageUploader(ABC):
    """Interface for image upload backends."""

    @abstractmethod
    def upload(self, local_path: str) -> str:
        """Upload image and return public URL."""
        pass
```

**S3 Implementation (Production):**
```python
class S3ImageUploader(ImageUploader):
    """Upload images to AWS S3 with public access."""

    def __init__(self, bucket_name: str, prefix: str = "datadog/"):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        self.prefix = prefix

    def upload(self, local_path: str) -> str:
        # Use fixed key to overwrite (simpler than updating dashboard)
        filename = Path(local_path).name
        key = f"{self.prefix}{filename}"

        self.s3.upload_file(
            local_path,
            self.bucket,
            key,
            ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
        )

        url = f"https://{self.bucket}.s3.amazonaws.com/{key}"
        return url
```

**Imgur Implementation (POC/Testing):**
```python
class ImgurUploader(ImageUploader):
    """Upload images to Imgur (simpler, no AWS setup)."""

    def __init__(self, client_id: str):
        self.client_id = client_id

    def upload(self, local_path: str) -> str:
        url = "https://api.imgur.com/3/image"
        headers = {"Authorization": f"Client-ID {self.client_id}"}

        with open(local_path, 'rb') as f:
            response = requests.post(
                url,
                headers=headers,
                files={'image': f}
            )

        response.raise_for_status()
        return response.json()['data']['link']
```

#### 3. Health Grid Implementation

```python
class HealthGridImageCollector(ImageCollector):
    """Generate health grid heatmap showing FoM values over 7 days."""

    # Metrics to display (rows)
    METRICS = [
        ("Tests Passing", "health.ci.tests_passing.fom"),
        ("Failing Workflows", "health.ci.failing_workflows.fom"),
        ("Hotfix Count", "health.ci.hotfix_count.fom"),
        ("Revert Count", "health.ci.revert_count.fom"),
        ("CI Duration P90", "health.ci.duration_p90.fom"),
        ("Stale PRs", "health.ci.stale_prs.fom"),
        ("PR Cycle Time", "health.ci.pr_cycle_time.fom"),
    ]

    def fetch_source_data(self) -> dict:
        """Fetch 7 days of FoM data for all metrics."""
        data = {}

        for display_name, metric_name in self.METRICS:
            daily_values = []

            # Query each of the last 7 days
            for day_offset in range(-6, 1):  # -6d to today (0)
                value = self.dd_client.query_metric_at_time(
                    metric_name,
                    aggregation="avg",
                    days_ago=abs(day_offset)
                )
                daily_values.append(value if value is not None else float('nan'))

            data[display_name] = daily_values

        return data

    def generate_image(self, data: dict) -> str:
        """Generate heatmap using seaborn."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Convert to 2D array (metrics × days)
        metric_names = list(data.keys())
        values = np.array([data[m] for m in metric_names])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Generate heatmap
        sns.heatmap(
            values,
            cmap="RdYlGn",  # Red → Yellow → Green
            vmin=0.0,
            vmax=1.0,
            annot=True,  # Show values in cells
            fmt=".2f",
            cbar_kws={"label": "Figure of Merit (FoM)"},
            xticklabels=["-6d", "-5d", "-4d", "-3d", "-2d", "-1d", "Today"],
            yticklabels=metric_names,
            linewidths=1,
            linecolor='white',
            square=False,
            ax=ax
        )

        # Styling
        ax.set_title(
            "System Health Rollup - 7 Day FoM Grid",
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel("Day", fontsize=14, fontweight='bold')
        ax.set_ylabel("Metric", fontsize=14, fontweight='bold')

        # Add threshold lines (optional)
        # ax.axhline(y=0, color='red', linewidth=2, alpha=0.3)  # Critical threshold

        plt.tight_layout()

        # Save to file
        output_path = "/tmp/health_grid_latest.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path
```

---

## Implementation Phases

### Phase 1: Proof of Concept (2 hours)

**Goal**: Validate the approach with a working prototype

**Tasks**:
1. Create `ImageCollector` base class
2. Implement `ImgurUploader` (simplest)
3. Implement `HealthGridImageCollector`
4. Test locally:
   ```bash
   python -m devops.datadog.collectors.health_grid_image
   # Generates image, uploads to Imgur, prints URL
   ```
5. Manually add image widget to dashboard
6. Verify image displays correctly

**Success Criteria**:
- ✅ Image generates successfully
- ✅ Image uploads to Imgur
- ✅ Image displays on Datadog dashboard
- ✅ Visual matches original spec better than widget grid

### Phase 2: Productionize (4 hours)

**Goal**: Make it production-ready with S3 and automation

**Tasks**:
1. Implement `S3ImageUploader`
2. Add AWS credentials to secrets manager
3. Create Helm chart for CronJob:
   ```yaml
   collectors:
     health_grid_image:
       enabled: true
       schedule: "0 * * * *"  # Hourly
       image_uploader: s3
       s3_bucket: softmax-datadog-images
   ```
4. Add error handling and retries
5. Add tests for image generation
6. Update documentation

**Success Criteria**:
- ✅ Runs automatically every hour
- ✅ Images stored in S3 with proper permissions
- ✅ Dashboard updates automatically (fixed URL)
- ✅ Monitoring/alerts for failures

### Phase 3: Generalize (6 hours)

**Goal**: Support multiple image-based visualizations

**Tasks**:
1. Extract `ImageCollector` to `common/base_image.py`
2. Create additional visualizations:
   - Training metrics trend chart
   - Eval metrics comparison
   - CI/CD velocity dashboard
3. Add configuration system:
   ```yaml
   image_collectors:
     health_grid:
       type: heatmap
       metrics: [...]
       days: 7
     training_trends:
       type: line_chart
       metrics: [...]
       timerange: 30d
   ```
4. Support multiple output formats (PNG, SVG)
5. Add caching to avoid regenerating identical images

---

## Technical Decisions

### 1. Image Storage: S3 vs Imgur

**Recommendation: S3 for production, Imgur for POC**

| Factor | S3 | Imgur |
|--------|----|----|
| Setup complexity | Medium (AWS config) | Low (API key) |
| Cost | ~$0.02/GB/month | Free (public) |
| Control | Full | Limited |
| Privacy | Private bucket option | Public only |
| Reliability | 99.99% SLA | Best effort |
| Integration | Native AWS | External API |

**Decision**:
- POC: Imgur (faster to test)
- Production: S3 (better control, already using AWS)

### 2. Fixed URL vs Dynamic URL

**Recommendation: Fixed URL (overwrite S3 key)**

**Option A: Fixed URL (Overwrite)**
```python
# Always upload to same key
s3.upload_file(local_path, bucket, "datadog/health_grid.png")
# URL never changes: https://bucket.s3.../health_grid.png
# Dashboard references same URL forever
```
✅ Pro: No dashboard updates needed
❌ Con: CDN caching issues

**Option B: Dynamic URL (Timestamp)**
```python
# Upload with timestamp
key = f"datadog/health_grid_{int(time.time())}.png"
s3.upload_file(local_path, bucket, key)
# URL changes each time
# Must update dashboard JSON with new URL
```
✅ Pro: No caching issues
❌ Con: Must update dashboard after each upload

**Decision**: Option A (Fixed URL) with cache-busting via query params if needed

### 3. Update Frequency

**Recommendation: Hourly (aligned with FoM collector)**

Options:
- **Every 15 min**: More real-time, but FoM data doesn't change that often
- **Hourly**: Matches FoM collector schedule
- **Every 6 hours**: Less resource usage, still fresh enough

**Decision**: Hourly
- Matches FoM collector (source data updates hourly)
- Good balance of freshness vs resource usage
- Can increase frequency later if needed

### 4. Dependencies

New Python packages needed:
```toml
[project.dependencies]
matplotlib = ">=3.8.0"
seaborn = ">=0.13.0"
pillow = ">=10.0.0"  # Image processing
boto3 = ">=1.34.0"   # S3 upload (already have this)
```

---

## Dashboard Integration

### Image Widget Configuration

```json
{
  "type": "image",
  "url": "https://softmax-datadog-images.s3.amazonaws.com/datadog/health_grid.png",
  "sizing": "fit",
  "margin": "small",
  "has_background": false,
  "has_border": false,
  "vertical_align": "top",
  "horizontal_align": "center",
  "url_dark_theme": "https://...health_grid_dark.png"  // Optional: dark mode variant
}
```

### Dashboard Template

```python
# In scripts/generate_health_dashboard.py
def create_image_widget(image_url: str) -> dict:
    return {
        "definition": {
            "type": "image",
            "url": image_url,
            "sizing": "fit",
            "margin": "small",
        },
        "layout": {
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 8,
        }
    }
```

---

## Comparison: Widget Grid vs Image

| Aspect | Widget Grid (Current) | Image Collector (Proposed) |
|--------|-----------------------|----------------------------|
| **Visual Control** | Limited (Datadog widgets) | Complete (matplotlib) |
| **Interactivity** | ✅ Hover, click, drill-down | ❌ Static image |
| **Real-time** | ✅ Live queries (~1 min) | ❌ Updates on schedule (hourly) |
| **Customization** | ❌ 12-col grid, fixed layouts | ✅ Pixel-perfect, any layout |
| **Maintenance** | Medium (64 widget JSON) | Low (Python code) |
| **Dashboard size** | Large (64 widgets) | Small (1 widget) |
| **Time selector** | ✅ Works | ❌ Fixed time range |
| **Annotations** | Limited | ✅ Any matplotlib feature |
| **Dependencies** | None | matplotlib, seaborn, S3 |
| **Complexity** | Low | Medium |

---

## Open Questions

1. **How to query Datadog for specific day?**
   - Current: `query_metric()` with time range
   - Need: Query metric value at specific timestamp
   - Solution: Extend `DatadogClient` with `query_metric_at_time(metric, days_ago)`

2. **Image refresh in browser?**
   - Fixed URL: Browser may cache
   - Solutions:
     - Add cache-busting query param: `?t=<timestamp>`
     - Set S3 Cache-Control headers: `max-age=300`
     - Use CloudFront invalidation

3. **What if image generation fails?**
   - Fallback: Show previous image (keep last good image)
   - Alert: Emit metric `health_grid_image.generation_success = 0`
   - Dashboard: Add backup static image widget

4. **Dark mode support?**
   - Generate two images: light and dark
   - Use `url_dark_theme` parameter in image widget
   - Or: Use transparent background, let Datadog handle theme

5. **Multiple visualizations?**
   - One collector per visualization type
   - Or: Single collector, multiple outputs
   - Recommendation: Separate collectors for clarity

---

## Next Steps

### Immediate (Now)
1. ✅ Get user feedback on this plan
2. Decide: POC now or later?
3. Choose: S3 or Imgur for POC?

### If Proceeding with POC
1. Create `devops/datadog/common/base_image.py`
2. Create `devops/datadog/common/image_uploaders.py`
3. Create `devops/datadog/collectors/health_grid_image/`
4. Test locally
5. Compare with widget grid
6. Decide which approach to use long-term

### Future Enhancements
- Multiple visualization types (trends, comparisons, distributions)
- Interactive elements (via plotly → static image)
- Automated A/B testing (show both approaches, track which users prefer)
- Dashboard generator that creates both widget and image variants

---

## Recommendation

**Try the POC** (2 hours investment):
1. Use Imgur for quick validation
2. Generate one heatmap image
3. Compare side-by-side with widget grid
4. Decide based on actual results

**Decision criteria**:
- If image looks significantly better → proceed to Phase 2
- If widget grid is good enough → stick with it
- Can always keep both (widget for interactivity, image for presentations)

**My prediction**: The image will look cleaner and more professional, but you'll miss the interactivity. Consider hybrid: widget grid for daily use, image for executive dashboards/reports.

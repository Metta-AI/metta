# Datadog Widget & API Reference

Reference for Datadog dashboard widgets, powerpacks, and APIs - optimized for programmatic use.

## Official Documentation Links

### Main Dashboard Documentation
- **Dashboard Overview**: https://docs.datadoghq.com/dashboards/
- **Dashboard JSON Schema**: https://docs.datadoghq.com/dashboards/graphing_json/
- **Dashboard API**: https://docs.datadoghq.com/api/latest/dashboards/

### Widget Documentation by Type

**Graph Widgets:**
- **Timeseries**: https://docs.datadoghq.com/dashboards/widgets/timeseries/
- **Top List**: https://docs.datadoghq.com/dashboards/widgets/top_list/
- **Query Value**: https://docs.datadoghq.com/dashboards/widgets/query_value/
- **Change**: https://docs.datadoghq.com/dashboards/widgets/change/
- **Heatmap**: https://docs.datadoghq.com/dashboards/widgets/heatmap/
- **Distribution**: https://docs.datadoghq.com/dashboards/widgets/distribution/
- **Table**: https://docs.datadoghq.com/dashboards/widgets/table/
- **Treemap**: https://docs.datadoghq.com/dashboards/widgets/treemap/
- **Pie Chart**: https://docs.datadoghq.com/dashboards/widgets/pie_chart/
- **Scatter Plot**: https://docs.datadoghq.com/dashboards/widgets/scatter_plot/
- **Geomap**: https://docs.datadoghq.com/dashboards/widgets/geomap/
- **Funnel**: https://docs.datadoghq.com/dashboards/widgets/funnel_analysis/
- **List**: https://docs.datadoghq.com/dashboards/widgets/list/

**Monitoring & Alerting:**
- **Alert Graph**: https://docs.datadoghq.com/dashboards/widgets/alert_graph/
- **Alert Value**: https://docs.datadoghq.com/dashboards/widgets/alert_value/
- **Monitor Summary**: https://docs.datadoghq.com/dashboards/widgets/monitor_summary/
- **Check Status**: https://docs.datadoghq.com/dashboards/widgets/check_status/
- **SLO Widget**: https://docs.datadoghq.com/dashboards/widgets/slo/
- **SLO List**: https://docs.datadoghq.com/dashboards/widgets/slo_list/

**Logs & Events:**
- **Event Stream**: https://docs.datadoghq.com/dashboards/widgets/event_stream/
- **Event Timeline**: https://docs.datadoghq.com/dashboards/widgets/event_timeline/
- **Log Stream**: https://docs.datadoghq.com/dashboards/widgets/log_stream/

**Infrastructure:**
- **Host Map**: https://docs.datadoghq.com/dashboards/widgets/hostmap/
- **Service Summary**: https://docs.datadoghq.com/dashboards/widgets/service_summary/
- **Topology Map**: https://docs.datadoghq.com/dashboards/widgets/topology_map/

**Layout & Documentation:**
- **Group**: https://docs.datadoghq.com/dashboards/widgets/group/
- **Note**: https://docs.datadoghq.com/dashboards/widgets/note/
- **Image**: https://docs.datadoghq.com/dashboards/widgets/image/
- **Iframe**: https://docs.datadoghq.com/dashboards/widgets/iframe/
- **Free Text**: https://docs.datadoghq.com/dashboards/widgets/free_text/

### Powerpacks
- **Powerpacks Overview**: https://docs.datadoghq.com/dashboards/powerpacks/
- **Creating Custom Powerpacks**: https://docs.datadoghq.com/dashboards/powerpacks/#create-powerpacks

### API Documentation
- **Create Dashboard**: https://docs.datadoghq.com/api/latest/dashboards/#create-a-new-dashboard
- **Update Dashboard**: https://docs.datadoghq.com/api/latest/dashboards/#update-a-dashboard
- **Get Dashboard**: https://docs.datadoghq.com/api/latest/dashboards/#get-a-dashboard
- **List Dashboards**: https://docs.datadoghq.com/api/latest/dashboards/#get-all-dashboards
- **Delete Dashboard**: https://docs.datadoghq.com/api/latest/dashboards/#delete-a-dashboard

### Query Language & Functions
- **Query Language**: https://docs.datadoghq.com/dashboards/querying/
- **Functions Reference**: https://docs.datadoghq.com/dashboards/functions/
- **Arithmetic**: https://docs.datadoghq.com/dashboards/functions/arithmetic/
- **Rate**: https://docs.datadoghq.com/dashboards/functions/rate/
- **Smoothing**: https://docs.datadoghq.com/dashboards/functions/smoothing/
- **Rollup**: https://docs.datadoghq.com/dashboards/functions/rollup/
- **Beta**: https://docs.datadoghq.com/dashboards/functions/beta/

---

## Widget Categories

### Graphs (Data Visualization)
- **timeseries** - Line/area/bar charts over time
- **toplist** - Ranked list of values
- **query_value** - Single number display
- **change** - Period-over-period comparison
- **heatmap** - Color-coded density visualization
- **distribution** - Histogram/percentile distribution
- **query_table** - Multi-column structured data
- **treemap** - Hierarchical data visualization
- **pie_chart** - Pie chart
- **scatter_plot** - X/Y scatter plot
- **geomap** - Geographic map
- **funnel** - Funnel analysis
- **list** - List widget

### Monitoring & Alerts
- **alert_graph** - Graph with alert thresholds
- **alert_value** - Monitor status with value
- **monitor_summary** - Overview of monitor states
- **check_status** - Service check status
- **slo** - Service level objective tracking
- **slo_list** - List of SLOs

### Logs & Events
- **event_stream** - Event timeline feed
- **event_timeline** - Condensed event view
- **log_stream** - Live log feed

### Infrastructure
- **hostmap** - Host infrastructure map
- **service_summary** - APM service summary
- **topology_map** - Service topology visualization

### Layout & Documentation
- **group** - Container for organizing widgets
- **note** - Markdown text and formatting
- **image** - Embedded images
- **iframe** - Embedded iframe
- **free_text** - Simple text labels

---

## Widget JSON Structure

All widgets follow this basic structure:

```json
{
  "id": 123456789,  // Auto-generated (optional for new widgets)
  "definition": {
    "type": "timeseries",  // Widget type (see categories above)
    "title": "Widget Title",
    "requests": [
      {
        "queries": [
          {
            "data_source": "metrics",  // or "logs", "apm", "rum", etc.
            "name": "query1",
            "query": "avg:metric.name{*}"
          }
        ],
        "formulas": [
          {
            "formula": "query1"
          }
        ]
      }
    ]
  },
  "layout": {  // Only for "free" layout type dashboards
    "x": 0,
    "y": 0,
    "width": 4,
    "height": 2
  }
}
```

### Common Widget Options

**Title Options:**
```json
{
  "title": "Widget Title",
  "title_size": "16",      // Font size
  "title_align": "left"    // "left", "center", "right"
}
```

**Legend Options:**
```json
{
  "show_legend": true,
  "legend_layout": "auto",  // "auto", "horizontal", "vertical"
  "legend_columns": ["avg", "min", "max", "value", "sum"]
}
```

**Style Options (Timeseries):**
```json
{
  "style": {
    "palette": "dog_classic",     // Color palette
    "line_type": "solid",         // "solid", "dashed", "dotted"
    "line_width": "normal",       // "thin", "normal", "thick"
  },
  "display_type": "line"          // "line", "area", "bars"
}
```

**Markers (Thresholds):**
```json
{
  "markers": [
    {
      "label": "Threshold label",
      "value": "y = 100",
      "display_type": "error dashed"  // "info", "warning", "error", "ok"
    }
  ]
}
```

---

## Query Patterns

### Basic Metric Query
```
avg:metric.name{tag:value}
```

### Aggregation by Tag
```
avg:metric.name{*} by {host}
```

### Multiple Tags
```
avg:metric.name{env:prod,service:api} by {host}
```

### Time Aggregation (Rate)
```
sum:metric.count{*}.as_rate()
```

### Top N Query
```
top(avg:metric.name{*} by {host}, 10, "mean", "desc")
```

### Arithmetic
```
avg:metric.a{*} / avg:metric.b{*}
```

### Functions
```
avg:metric.name{*}.rollup(avg, 60)           // 60-second rollup
avg:metric.name{*}.rollup(sum, 3600)         // 1-hour rollup
avg:metric.name{*}.fill(zero)                // Fill gaps with zero
moving_rollup(avg:metric.name{*}, 300, "avg") // 5-minute moving average
```

---

## Data Sources

Widgets can query from multiple data sources:

- **metrics** - Datadog metrics
- **logs** - Log data
- **apm** - APM/trace data
- **apm_resource_stats** - APM resource statistics
- **rum** - Real User Monitoring
- **network** - Network performance
- **process** - Process metrics
- **security** - Security signals
- **events** - Event data

---

## Powerpack Use Cases

Powerpacks are pre-configured widget collections for specific technologies:

### Infrastructure
- Kubernetes cluster monitoring
- Docker container monitoring
- AWS/Azure/GCP service monitoring
- Host and system monitoring

### Databases
- PostgreSQL, MySQL, MongoDB performance
- Redis cache monitoring
- Elasticsearch cluster health

### Application Services
- NGINX/Apache web server
- Kafka/RabbitMQ message queues
- Node.js/Python/Java applications

### CI/CD
- GitHub Actions
- Jenkins pipelines
- CircleCI

---

## Common Widget Implementation Patterns

### Single Value Display
```jsonnet
widgets.queryValue(
  title='Current Value',
  query='avg:metric.name{*}',
  options={
    precision: 2,
    custom_unit: 'ms',
  }
)
```

### Timeseries with Threshold
```jsonnet
widgets.timeseries(
  title='Response Time',
  query='avg:http.response_time{*}',
  options={
    markers: [{
      label: 'SLA threshold',
      value: 'y = 100',
      display_type: 'warning dashed',
    }],
  }
)
```

### Top List
```jsonnet
widgets.toplist(
  title='Top 10 by CPU',
  query='top(avg:system.cpu.user{*} by {host}, 10, "mean", "desc")'
)
```

### Multi-Column Table
```jsonnet
widgets.table(
  title='System Metrics',
  queries=[
    { query: 'avg:system.cpu.user{*} by {host}', alias: 'CPU %' },
    { query: 'avg:system.mem.used{*} by {host}', alias: 'Memory' },
  ]
)
```

---

## Programmatic Access

### List Metrics
```bash
./scripts/list_metrics.py
./scripts/list_metrics.py --search=cpu
```

### Export Dashboard
```bash
./scripts/export_dashboard.py <dashboard-id> > dashboard.json
```

### Push Dashboard
```bash
./scripts/push_dashboard.py dashboard.json
```

### Batch Export
```bash
./scripts/batch_export.py
```

---

## Extending Our Library

To add new widget types to `lib/widgets.libsonnet`:

1. **Find Widget Documentation**: Use links above to find official widget docs
2. **Export Example**: Use `./scripts/export_dashboard.py` to see real JSON
3. **Extract Pattern**: Identify the `definition` structure
4. **Create Function**: Add to `lib/widgets.libsonnet` with parameterization
5. **Test**: Build dashboard with `metta datadog dashboard build` and verify JSON

### Example: Adding New Widget Type
```jsonnet
// In lib/widgets.libsonnet
{
  newWidgetType(title, query, options={}):: {
    definition: {
      type: 'new_widget_type',
      title: title,
      // ... widget-specific fields
      requests: [
        {
          queries: [{ query: query }],
          formulas: [{ formula: 'query1' }],
        }
      ],
    },
  },
}
```

---

Last updated: 2025-10-22

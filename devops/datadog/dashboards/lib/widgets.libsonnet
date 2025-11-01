// Widget library - Primitive widget builders for Datadog dashboards
// Inspired by Grafana's Grafonnet
//
// This library provides LLM-friendly, progressively-disclosed widget builders.
// Each widget follows the pattern: simple string query → options object → advanced object
//
// Documentation conventions:
// - @widget: Widget type name
// - @purpose: What this widget is used for
// - @simple: Simplest usage with just title and query
// - @options: Common customization options
// - @advanced: Full control with complex queries/formulas
// - @enum: Valid values for an option
// - @related: Similar or related widgets
// - @docs: Link to Datadog documentation
//
// Example search keywords for LLMs:
// - "timeseries chart line graph" → timeseries()
// - "single number metric gauge" → queryValue()
// - "ranked list top n" → toplist()
// - "text header markdown note" → note()
// - "heatmap density" → heatmap()

{
  // ========== TIMESERIES WIDGET ==========
  //
  // @widget: timeseries
  // @purpose: Display metric trends over time as lines, bars, or areas
  // @use_cases: CPU usage, memory trends, request rates, latency over time
  //
  // @simple: widgets.timeseries('CPU Usage', 'avg:system.cpu{*}')
  //
  // @options: Customize appearance and behavior
  //   - display_type: 'line' | 'bars' | 'area' (default: 'line')
  //   - palette: 'dog_classic' | 'warm' | 'cool' | 'purple' | 'orange' | 'gray' (default: 'dog_classic')
  //   - line_type: 'solid' | 'dashed' | 'dotted' (default: 'solid')
  //   - line_width: 'thin' | 'normal' | 'thick' (default: 'normal')
  //   - show_legend: true | false (default: false)
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //   - markers: Array of reference lines (e.g., SLO thresholds)
  //
  // @example_moderate:
  //   widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  //     display_type: 'area',
  //     palette: 'warm',
  //     show_legend: true,
  //   })
  //
  // @example_advanced:
  //   widgets.timeseries('CPU Usage', 'avg:system.cpu{*}', {
  //     markers: [
  //       { label: 'Warning', value: 'y = 80', display_type: 'warning dashed' },
  //       { label: 'Critical', value: 'y = 95', display_type: 'error dashed' },
  //     ],
  //   })
  //
  // @related: queryValue, heatmap, distribution
  // @docs: https://docs.datadoghq.com/dashboards/widgets/timeseries/
  //
  timeseries(title, query, options={}):: {
    definition: {
      type: 'timeseries',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      show_legend: if std.objectHas(options, 'show_legend') then options.show_legend else false,
      legend_layout: 'auto',
      legend_columns: ['avg', 'min', 'max', 'value', 'sum'],
      time: {},
      requests: [
        {
          formulas: [{ formula: 'query1' }],
          queries: [
            {
              data_source: 'metrics',
              name: 'query1',
              query: query,
            },
          ],
          response_format: 'timeseries',
          style: {
            order_by: 'values',
            palette: if std.objectHas(options, 'palette') then options.palette else 'dog_classic',
            line_type: if std.objectHas(options, 'line_type') then options.line_type else 'solid',
            line_width: if std.objectHas(options, 'line_width') then options.line_width else 'normal',
          },
          display_type: if std.objectHas(options, 'display_type') then options.display_type else 'line',
        },
      ],
      [if std.objectHas(options, 'markers') then 'markers']: options.markers,
    },
  },

  // ========== QUERY VALUE WIDGET ==========
  //
  // @widget: queryValue
  // @purpose: Display a single metric value as a large number (gauge/counter)
  // @use_cases: Current request count, active users, error rate, uptime percentage
  //
  // @simple: widgets.queryValue('Active Users', 'sum:app.users.active{*}')
  //
  // @options: Customize display and aggregation
  //   - precision: Number of decimal places (default: 2)
  //   - aggregator: 'avg' | 'sum' | 'min' | 'max' | 'last' (default: 'avg')
  //   - autoscale: true (default) | false - automatic unit scaling (e.g., 1000 → 1K)
  //   - custom_unit: Custom unit string (e.g., 'req/s', '%', 'ms')
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //
  // @example_moderate:
  //   widgets.queryValue('Error Rate', 'avg:app.errors.rate{*}', {
  //     precision: 1,
  //     custom_unit: '%',
  //     autoscale: false,
  //   })
  //
  // @example_advanced:
  //   widgets.queryValue('Total Requests', 'sum:app.requests{*}', {
  //     precision: 0,
  //     aggregator: 'sum',
  //     autoscale: true,
  //   })
  //
  // @related: timeseries, change, toplist
  // @docs: https://docs.datadoghq.com/dashboards/widgets/query_value/
  //
  queryValue(title, query, options={}):: {
    definition: {
      type: 'query_value',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      precision: if std.objectHas(options, 'precision') then options.precision else 2,
      requests: [
        {
          formulas: [{ formula: 'query1' }],
          queries: [
            {
              data_source: 'metrics',
              name: 'query1',
              query: query,
              aggregator: if std.objectHas(options, 'aggregator') then options.aggregator else 'avg',
            },
          ],
          response_format: 'scalar',
        },
      ],
      autoscale: if std.objectHas(options, 'autoscale') then options.autoscale else true,
      [if std.objectHas(options, 'custom_unit') then 'custom_unit']: options.custom_unit,
    },
  },

  // ========== TOPLIST WIDGET ==========
  //
  // @widget: toplist
  // @purpose: Display ranked list of metric values (top N)
  // @use_cases: Top hosts by CPU, busiest services, highest error rates
  //
  // @simple: widgets.toplist('Top Hosts by CPU', 'avg:system.cpu{*} by {host}')
  //
  // @options: Customize display
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //
  // @example_moderate:
  //   widgets.toplist('Busiest Services', 'sum:requests{*} by {service}', {
  //     title_size: '18',
  //   })
  //
  // @related: queryValue, table, heatmap
  // @docs: https://docs.datadoghq.com/dashboards/widgets/top_list/
  //
  toplist(title, query, options={}):: {
    definition: {
      type: 'toplist',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      requests: [
        {
          formulas: [{ formula: 'query1' }],
          queries: [
            {
              data_source: 'metrics',
              name: 'query1',
              query: query,
            },
          ],
          response_format: 'scalar',
        },
      ],
    },
  },

  // ========== NOTE WIDGET ==========
  //
  // @widget: note
  // @purpose: Display text, markdown, or section headers
  // @use_cases: Dashboard titles, section dividers, documentation, alerts
  //
  // @simple: widgets.note('## Performance Metrics')
  //
  // @options: Customize appearance
  //   - background_color: 'white' (default) | 'blue' | 'purple' | 'gray' | 'yellow' | 'red'
  //   - font_size: '14' (default) | '16' | '18' | '20'
  //   - text_align: 'left' (default) | 'center' | 'right'
  //   - vertical_align: 'top' (default) | 'center' | 'bottom'
  //   - show_tick: false (default) | true - show pointer arrow
  //   - tick_pos: '50%' (default) - position of pointer arrow
  //   - tick_edge: 'left' (default) | 'right' | 'top' | 'bottom'
  //   - has_padding: true (default) | false
  //
  // @example_moderate:
  //   widgets.note('## System Health', {
  //     background_color: 'blue',
  //     font_size: '18',
  //     text_align: 'center',
  //   })
  //
  // @example_advanced:
  //   widgets.note('**Alert**: High CPU usage detected\n\nCheck logs for details', {
  //     background_color: 'red',
  //     show_tick: true,
  //     tick_edge: 'left',
  //   })
  //
  // @related: group (for organizing widgets)
  // @docs: https://docs.datadoghq.com/dashboards/widgets/note/
  //
  note(content, options={}):: {
    definition: {
      type: 'note',
      content: content,
      background_color: if std.objectHas(options, 'background_color') then options.background_color else 'white',
      font_size: if std.objectHas(options, 'font_size') then options.font_size else '14',
      text_align: if std.objectHas(options, 'text_align') then options.text_align else 'left',
      vertical_align: if std.objectHas(options, 'vertical_align') then options.vertical_align else 'top',
      show_tick: if std.objectHas(options, 'show_tick') then options.show_tick else false,
      tick_pos: if std.objectHas(options, 'tick_pos') then options.tick_pos else '50%',
      tick_edge: if std.objectHas(options, 'tick_edge') then options.tick_edge else 'left',
      has_padding: if std.objectHas(options, 'has_padding') then options.has_padding else true,
    },
  },

  // ========== HEATMAP WIDGET ==========
  //
  // @widget: heatmap
  // @purpose: Display metric distribution as a density plot (color intensity)
  // @use_cases: Latency distributions, request duration patterns, host metrics
  //
  // @simple: widgets.heatmap('Request Latency', 'avg:request.duration{*} by {host}')
  //
  // @options: Customize display
  //   - palette: 'dog_classic' (default) | 'warm' | 'cool' | 'purple' | 'orange'
  //   - show_legend: true (default) | false
  //   - legend_size: '0' (default) | '2' | '4' | '8'
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //   - include_zero: true (default) | false - Y-axis starts at zero
  //   - scale: 'linear' (default) | 'log' | 'sqrt'
  //
  // @example_moderate:
  //   widgets.heatmap('Response Time Distribution', 'avg:response.time{*}', {
  //     palette: 'warm',
  //     show_legend: true,
  //   })
  //
  // @related: timeseries, distribution, toplist
  // @docs: https://docs.datadoghq.com/dashboards/widgets/heatmap/
  //
  heatmap(title, query, options={}):: {
    definition: {
      type: 'heatmap',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      show_legend: if std.objectHas(options, 'show_legend') then options.show_legend else true,
      legend_size: if std.objectHas(options, 'legend_size') then options.legend_size else '0',
      time: {},
      requests: [
        {
          formulas: [{ formula: 'query1' }],
          queries: [
            {
              data_source: 'metrics',
              name: 'query1',
              query: query,
            },
          ],
          response_format: 'timeseries',
          style: {
            palette: if std.objectHas(options, 'palette') then options.palette else 'dog_classic',
          },
        },
      ],
      yaxis: {
        include_zero: if std.objectHas(options, 'include_zero') then options.include_zero else true,
        scale: if std.objectHas(options, 'scale') then options.scale else 'linear',
      },
    },
  },

  // ========== CHANGE WIDGET ==========
  //
  // @widget: change
  // @purpose: Display metric change compared to a previous time period
  // @use_cases: Day-over-day changes, week-over-week trends, growth metrics
  //
  // @simple: widgets.change('Daily Active Users Change', 'avg:users.active{*}')
  //
  // @options: Customize comparison
  //   - compare_to: 'hour_before' (default) | 'day_before' | 'week_before' | 'month_before'
  //   - increase_good: true (default) | false - green for increase vs decrease
  //   - order_by: 'change' (default) | 'name' | 'present' | 'past'
  //   - order_dir: 'desc' (default) | 'asc'
  //   - show_present: true (default) | false - show current value
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //
  // @example_moderate:
  //   widgets.change('Weekly Revenue Growth', 'sum:revenue{*}', {
  //     compare_to: 'week_before',
  //     increase_good: true,
  //   })
  //
  // @example_advanced:
  //   widgets.change('Error Rate Change', 'avg:errors.rate{*}', {
  //     compare_to: 'day_before',
  //     increase_good: false,  // Increase in errors is bad
  //     order_by: 'change',
  //   })
  //
  // @related: queryValue, timeseries
  // @docs: https://docs.datadoghq.com/dashboards/widgets/change/
  //
  change(title, query, options={}):: {
    definition: {
      type: 'change',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      time: {},
      requests: [
        {
          formulas: [{ formula: 'query1' }],
          queries: [
            {
              data_source: 'metrics',
              name: 'query1',
              query: query,
            },
          ],
          response_format: 'scalar',
          compare_to: if std.objectHas(options, 'compare_to') then options.compare_to else 'hour_before',
          increase_good: if std.objectHas(options, 'increase_good') then options.increase_good else true,
          order_by: if std.objectHas(options, 'order_by') then options.order_by else 'change',
          order_dir: if std.objectHas(options, 'order_dir') then options.order_dir else 'desc',
          show_present: if std.objectHas(options, 'show_present') then options.show_present else true,
        },
      ],
    },
  },

  // ========== DISTRIBUTION WIDGET ==========
  //
  // @widget: distribution
  // @purpose: Display histogram of metric distribution (APM/tracing focused)
  // @use_cases: Request latency histogram, span duration distribution
  //
  // @simple: widgets.distribution('Latency Distribution', 'avg:trace.duration{*}')
  //
  // @options: Customize APM query and display
  //   - stat: 'avg' (default) | 'p50' | 'p75' | 'p90' | 'p95' | 'p99' | 'max'
  //   - service: Service name filter (required for APM)
  //   - env: Environment filter
  //   - operation_name: Operation name filter
  //   - primary_tag_value: Primary tag value (default: '*')
  //   - palette: 'dog_classic' (default) | 'warm' | 'cool' | 'purple' | 'orange'
  //   - show_legend: false (default) | true
  //   - include_zero: true (default) | false - X-axis starts at zero
  //   - scale: 'linear' (default) | 'log' | 'sqrt'
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //
  // @example_moderate:
  //   widgets.distribution('Request Duration', 'avg:trace.duration{*}', {
  //     service: 'web-api',
  //     env: 'production',
  //     palette: 'warm',
  //   })
  //
  // @note: This widget is primarily for APM/tracing data
  // @related: heatmap, timeseries
  // @docs: https://docs.datadoghq.com/dashboards/widgets/distribution/
  //
  distribution(title, query, options={}):: {
    definition: {
      type: 'distribution',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      show_legend: if std.objectHas(options, 'show_legend') then options.show_legend else false,
      time: {},
      requests: [
        {
          query: {
            stat: if std.objectHas(options, 'stat') then options.stat else 'avg',
            data_source: 'apm_resource_stats',
            name: 'query1',
            service: if std.objectHas(options, 'service') then options.service else '',
            env: if std.objectHas(options, 'env') then options.env else '',
            primary_tag_value: if std.objectHas(options, 'primary_tag_value') then options.primary_tag_value else '*',
            operation_name: if std.objectHas(options, 'operation_name') then options.operation_name else '',
          },
          request_type: 'histogram',
          style: {
            palette: if std.objectHas(options, 'palette') then options.palette else 'dog_classic',
          },
        },
      ],
      xaxis: {
        include_zero: if std.objectHas(options, 'include_zero') then options.include_zero else true,
        scale: if std.objectHas(options, 'scale') then options.scale else 'linear',
      },
      yaxis: {
        include_zero: true,
        scale: 'linear',
      },
    },
  },

  // ========== TABLE WIDGET ==========
  //
  // @widget: table
  // @purpose: Display metrics in tabular format with multiple columns
  // @use_cases: Multi-metric comparisons, service health matrices, resource inventories
  //
  // @simple: widgets.table('Service Health', 'avg:service.requests{*} by {service}')
  //
  // @options: Customize table display
  //   - has_search_bar: 'auto' (default) | 'always' | 'never'
  //   - title_size: '16' (default) | '18' | '20'
  //   - title_align: 'left' (default) | 'center' | 'right'
  //
  // @example_moderate (single query):
  //   widgets.table('Top Services', 'avg:requests{*} by {service}', {
  //     has_search_bar: 'always',
  //   })
  //
  // @example_advanced (multiple queries with aliases):
  //   widgets.table('Service Metrics', [
  //     { query: 'avg:requests{*} by {service}', alias: 'Requests', aggregator: 'sum' },
  //     { query: 'avg:errors{*} by {service}', alias: 'Errors', aggregator: 'sum' },
  //     { query: 'avg:latency{*} by {service}', alias: 'Latency', aggregator: 'avg' },
  //   ])
  //
  // @note: Can accept single query string or array of query objects
  // @related: toplist, queryValue
  // @docs: https://docs.datadoghq.com/dashboards/widgets/table/
  //
  table(title, queries, options={}):: {
    definition: {
      type: 'query_table',
      title: title,
      title_size: if std.objectHas(options, 'title_size') then options.title_size else '16',
      title_align: if std.objectHas(options, 'title_align') then options.title_align else 'left',
      time: {},
      requests: [
        {
          formulas: if std.isArray(queries) then [
            { formula: 'query%d' % (i + 1), alias: if std.objectHas(q, 'alias') then q.alias else null }
            for i in std.range(0, std.length(queries) - 1)
            for q in [queries[i]]
          ] else [{ formula: 'query1' }],
          queries: if std.isArray(queries) then [
            {
              data_source: 'metrics',
              name: 'query%d' % (i + 1),
              query: q.query,
              aggregator: if std.objectHas(q, 'aggregator') then q.aggregator else 'avg',
            }
            for i in std.range(0, std.length(queries) - 1)
            for q in [queries[i]]
          ] else [
            {
              data_source: 'metrics',
              name: 'query1',
              query: queries,
              aggregator: 'avg',
            },
          ],
          response_format: 'scalar',
        },
      ],
      has_search_bar: if std.objectHas(options, 'has_search_bar') then options.has_search_bar else 'auto',
    },
  },

  // ========== GROUP WIDGET ==========
  //
  // @widget: group
  // @purpose: Container for organizing related widgets into sections
  // @use_cases: Logical grouping, collapsible sections, visual organization
  //
  // @simple: widgets.group('Database Metrics', [widget1, widget2, widget3])
  //
  // @options: Customize group appearance
  //   - layout_type: 'ordered' (default) | 'free' - grid vs free positioning
  //   - background_color: 'vivid_blue' (default) | 'vivid_purple' | 'vivid_pink' | 'vivid_orange' | 'vivid_yellow' | 'vivid_green' | 'gray'
  //   - show_title: true (default) | false
  //
  // @example_moderate:
  //   widgets.group('API Metrics', [
  //     requestsWidget,
  //     latencyWidget,
  //     errorsWidget,
  //   ], {
  //     background_color: 'vivid_purple',
  //   })
  //
  // @example_advanced:
  //   widgets.group('Infrastructure', infrastructureWidgets, {
  //     layout_type: 'free',
  //     background_color: 'gray',
  //     show_title: true,
  //   })
  //
  // @related: note (for section headers)
  // @docs: https://docs.datadoghq.com/dashboards/widgets/group/
  //
  group(title, widgets, options={}):: {
    definition: {
      type: 'group',
      title: title,
      layout_type: if std.objectHas(options, 'layout_type') then options.layout_type else 'ordered',
      widgets: widgets,
      background_color: if std.objectHas(options, 'background_color') then options.background_color else 'vivid_blue',
      show_title: if std.objectHas(options, 'show_title') then options.show_title else true,
    },
  },
}

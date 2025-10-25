// Policy Evaluator APM Dashboard
// Monitors eval-orchestrator and eval-worker latency and throughput

local layouts = import '../lib/layouts.libsonnet';

// Manual APM timeseries widget helper (APM metrics don't work with standard widgets.timeseries)
local apmTimeseries(title, queries, options={}) = {
  definition: {
    title: title,
    title_size: '16',
    title_align: 'left',
    [if std.objectHas(options, 'show_legend') then 'show_legend']: options.show_legend,
    [if std.objectHas(options, 'legend_layout') then 'legend_layout']: options.legend_layout,
    [if std.objectHas(options, 'legend_columns') then 'legend_columns']: options.legend_columns,
    time: {},
    type: 'timeseries',
    requests: [
      {
        formulas: std.map(
          function(q) {
            [if std.objectHas(q, 'style') then 'style']: q.style,
            formula: q.name,
          },
          queries
        ),
        queries: std.map(
          function(q) {
            data_source: q.data_source,
            name: q.name,
            stat: q.stat,
            service: q.service,
            group_by: if std.objectHas(q, 'group_by') then q.group_by else [],
          } + (if std.objectHas(q, 'span_kind') then { span_kind: q.span_kind } else {})
            + (if std.objectHas(q, 'operation_mode') then { operation_mode: q.operation_mode } else {}),
          queries
        ),
        response_format: 'timeseries',
        style: {
          palette: if std.objectHas(options, 'palette') then options.palette else 'dog_classic',
          order_by: 'values',
          line_type: if std.objectHas(options, 'line_type') then options.line_type else 'solid',
          line_width: if std.objectHas(options, 'line_width') then options.line_width else 'normal',
        },
        display_type: if std.objectHas(options, 'display_type') then options.display_type else 'line',
      },
    ],
  },
};

layouts.grid(
  'Policy Evaluator',
  std.flattenArrays([
    // Row 1: Three APM metrics side by side
    layouts.row(0, [
      // Eval Orchestrator Latency (p90 and p99)
      apmTimeseries(
        'Eval Orchestrator Latency',
        [
          {
            data_source: 'apm_metrics',
            name: 'query1',
            stat: 'latency_p90',
            service: 'eval-orchestrator',
            group_by: [],
            style: {
              palette: 'classic',
              palette_index: 1,
            },
          },
          {
            data_source: 'apm_metrics',
            name: 'query2',
            stat: 'latency_p99',
            service: 'eval-orchestrator',
            group_by: [],
            style: {
              palette: 'purple',
              palette_index: 5,
            },
          },
        ],
        {
          show_legend: true,
          legend_layout: 'auto',
          legend_columns: ['avg', 'min', 'max', 'value', 'sum'],
        }
      ),

      // Eval Worker Latency (p90 by operation)
      apmTimeseries(
        'Eval Worker Latency by Operation',
        [
          {
            data_source: 'apm_metrics',
            name: 'query1',
            stat: 'latency_p90',
            service: 'eval-worker',
            span_kind: 'internal',
            operation_mode: 'union',
            group_by: ['operation_name'],
          },
        ]
      ),

      // Eval Worker Hits (request count)
      apmTimeseries(
        'Eval Worker Request Rate',
        [
          {
            data_source: 'apm_metrics',
            name: 'query1',
            stat: 'hits',
            service: 'eval-worker',
          },
        ],
        {
          show_legend: true,
          legend_layout: 'auto',
          legend_columns: ['avg', 'min', 'max', 'value', 'sum'],
        }
      ),
    ], height=4),
  ]),
  {
    id: 'gpk-2y2-9er',  // Preserve existing dashboard ID
    description: 'Policy evaluation service performance monitoring',
  }
)

// Widget library - Primitive widget builders for Datadog dashboards
// Inspired by Grafana's Grafonnet

{
  // Basic timeseries widget
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

  // Query value widget (single number display)
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

  // Top list widget
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

  // Note/markdown widget
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

  // Heatmap widget
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

  // Change widget (shows change over time period)
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

  // Distribution widget (histogram)
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

  // Table widget
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

  // Group widget (container for organizing widgets)
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

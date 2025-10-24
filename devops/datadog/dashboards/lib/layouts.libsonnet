// Dashboard layout templates for Datadog
// Provides three layout types: grid (dashboard), auto (timeboard), and free (screenboard)

{
  // Grid layout (Dashboard) - Widgets snap to grid
  // Default 12-column grid with automatic row placement
  grid(title, widgets, options={})::
    local base = {
      title: title,
      description: if std.objectHas(options, 'description') then options.description else '',
      layout_type: 'ordered',  // Grid layout
      template_variables: if std.objectHas(options, 'template_variables') then options.template_variables else [],
      widgets: widgets,
    };

    // Only include optional fields if provided
    base +
    (if std.objectHas(options, 'id') then { id: options.id } else {}) +
    (if std.objectHas(options, 'notify_list') then { notify_list: options.notify_list } else {}) +
    (if std.objectHas(options, 'reflow_type') then { reflow_type: options.reflow_type } else {}),

  // Automatic layout (Timeboard) - Fits browser automatically
  auto(title, widgets, options={})::
    local base = {
      title: title,
      description: if std.objectHas(options, 'description') then options.description else '',
      layout_type: 'ordered',  // Auto layout
      template_variables: if std.objectHas(options, 'template_variables') then options.template_variables else [],
      widgets: widgets,
    };

    // Only include optional fields if provided
    base +
    (if std.objectHas(options, 'id') then { id: options.id } else {}) +
    (if std.objectHas(options, 'is_read_only') then { is_read_only: options.is_read_only } else {}) +
    (if std.objectHas(options, 'notify_list') then { notify_list: options.notify_list } else {}),

  // Free layout (Screenboard) - Pixel-level precision
  free(title, widgets, options={})::
    local base = {
      title: title,
      description: if std.objectHas(options, 'description') then options.description else '',
      layout_type: 'free',  // Free positioning
      template_variables: if std.objectHas(options, 'template_variables') then options.template_variables else [],
      widgets: widgets,
    };

    // Only include optional fields if provided
    base +
    (if std.objectHas(options, 'id') then { id: options.id } else {}) +
    (if std.objectHas(options, 'is_read_only') then { is_read_only: options.is_read_only } else {}) +
    (if std.objectHas(options, 'notify_list') then { notify_list: options.notify_list } else {}),

  // Helper: Position widget at specific grid coordinates
  // Grid is 12 columns wide by default
  at(widget, x, y, width, height):: widget + {
    layout: {
      x: x,
      y: y,
      width: width,
      height: height,
    },
  },

  // Helper: Create a row of widgets with automatic column distribution
  // Widgets are evenly distributed across 12 columns
  row(y, widgets, height=3)::
    local colWidth = 12 / std.length(widgets);
    [
      widgets[i] + {
        layout: {
          x: i * colWidth,
          y: y,
          width: colWidth,
          height: height,
        },
      }
      for i in std.range(0, std.length(widgets) - 1)
    ],

  // Helper: Create a row with custom column widths
  // widths: array of column widths (must sum to 12 or less)
  rowCustom(y, widgets, widths, height=3)::
    assert std.length(widgets) == std.length(widths) : 'widgets and widths must have same length';
    local xPositions = std.foldl(
      function(acc, w) acc + [if std.length(acc) == 0 then 0 else acc[std.length(acc) - 1] + widths[std.length(acc) - 1]],
      widths,
      []
    );
    [
      widgets[i] + {
        layout: {
          x: xPositions[i],
          y: y,
          width: widths[i],
          height: height,
        },
      }
      for i in std.range(0, std.length(widgets) - 1)
    ],

  // Helper: Create a column of widgets
  column(x, startY, widgets, width=6, height=3):: [
    widgets[i] + {
      layout: {
        x: x,
        y: startY + (i * height),
        width: width,
        height: height,
      },
    }
    for i in std.range(0, std.length(widgets) - 1)
  ],

  // Helper: Full-width widget (12 columns)
  fullWidth(y, widget, height=3):: widget + {
    layout: {
      x: 0,
      y: y,
      width: 12,
      height: height,
    },
  },

  // Helper: Half-width widget (6 columns)
  halfWidth(y, widget, left=true, height=3):: widget + {
    layout: {
      x: if left then 0 else 6,
      y: y,
      width: 6,
      height: height,
    },
  },

  // Helper: Third-width widget (4 columns)
  thirdWidth(y, widget, position=0, height=3)::
    assert position >= 0 && position <= 2 : 'position must be 0, 1, or 2';
    widget + {
      layout: {
        x: position * 4,
        y: y,
        width: 4,
        height: height,
      },
    },

  // Helper: Quarter-width widget (3 columns)
  quarterWidth(y, widget, position=0, height=2)::
    assert position >= 0 && position <= 3 : 'position must be 0, 1, 2, or 3';
    widget + {
      layout: {
        x: position * 3,
        y: y,
        width: 3,
        height: height,
      },
    },

  // Helper: Create grid of widgets (e.g., 2x2, 3x3)
  // cols: number of columns (e.g., 2 for 2x2 grid)
  // rows: number of rows
  grid2d(startY, widgets, cols, rows, options={})::
    local colWidth = 12 / cols;
    local height = if std.objectHas(options, 'height') then options.height else 3;
    local spacing = if std.objectHas(options, 'spacing') then options.spacing else 0;
    [
      widgets[i] + {
        layout: {
          x: (i % cols) * colWidth,
          y: startY + (std.floor(i / cols) * (height + spacing)),
          width: colWidth,
          height: height,
        },
      }
      for i in std.range(0, std.length(widgets) - 1)
    ],

  // Helper: Create template variable
  templateVar(name, prefix, options={}):: {
    name: name,
    prefix: prefix,
    available_values: if std.objectHas(options, 'available_values') then options.available_values else [],
    default: if std.objectHas(options, 'default') then options.default else '*',
  },
}

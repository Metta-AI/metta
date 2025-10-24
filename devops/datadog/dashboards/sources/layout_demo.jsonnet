// Layout Demo Dashboard
// Demonstrates all layout helpers and positioning functions

local layouts = import '../lib/layouts.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

// Create some demo widgets
local demoQueryValue(title) = widgets.queryValue(
  title,
  'avg:system.cpu.user{*}',
  { precision: 0 }
);

local demoTimeseries(title) = widgets.timeseries(
  title,
  'avg:system.cpu.user{*}',
  { display_type: 'line' }
);

local demoNote(content) = widgets.note(
  content,
  { background_color: 'blue', font_size: '18', text_align: 'center' }
);

// Build dashboard using layout helpers
layouts.grid(
  'Layout Demo Dashboard',
  // Flatten all widget arrays into single array
  std.flattenArrays([
    // Row 1: Four equal-width widgets (quarter-width each)
    layouts.row(0, [
      demoQueryValue('Metric 1'),
      demoQueryValue('Metric 2'),
      demoQueryValue('Metric 3'),
      demoQueryValue('Metric 4'),
    ], height=2),

    // Row 2: Full-width section header
    [layouts.fullWidth(2, demoNote('## Timeseries Charts'), height=1)],

    // Row 3: Two half-width timeseries
    layouts.row(3, [
      demoTimeseries('CPU Usage'),
      demoTimeseries('Memory Usage'),
    ], height=3),

    // Row 4: Three equal-width widgets (third-width each)
    layouts.row(6, [
      demoTimeseries('Disk I/O'),
      demoTimeseries('Network'),
      demoTimeseries('Processes'),
    ], height=3),

    // Row 5: Custom widths - 4 cols, 4 cols, 4 cols
    layouts.rowCustom(
      9,
      [
        demoQueryValue('Wide Metric 1'),
        demoQueryValue('Wide Metric 2'),
        demoQueryValue('Wide Metric 3'),
      ],
      [4, 4, 4],  // Custom column widths
      height=2
    ),

    // Row 6: Section header
    [layouts.fullWidth(11, demoNote('## 2x2 Grid Layout'), height=1)],

    // Row 7-8: 2x2 grid of widgets
    layouts.grid2d(
      12,
      [
        demoTimeseries('Grid 1'),
        demoTimeseries('Grid 2'),
        demoTimeseries('Grid 3'),
        demoTimeseries('Grid 4'),
      ],
      2,  // cols
      2,  // rows
      { height: 3 }
    ),

    // Row 9: Manual positioning with at()
    [layouts.at(demoQueryValue('Custom Position'), x=0, y=18, width=3, height=2)],
    [layouts.at(demoQueryValue('Another Custom'), x=9, y=18, width=3, height=2)],
  ]),
  {
    description: 'Demonstrates all layout helpers: rows, columns, grids, and custom positioning',
  }
)

// CI/CD component library
// Widgets for continuous integration and deployment health

local widgets = import '../lib/widgets.libsonnet';

{
  // Tests passing on main branch
  testsPassingWidget()::
    widgets.timeseries(
      title='Tests are passing on main',
      query='avg:ci.tests_passing_on_main{source:softmax-system-health}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Unit-test jobs should be passing on main ',
            value: 'y = 1',
            display_type: 'info undefined',
          },
        ],
      }
    ),

  // Number of reverts in the last 7 days
  revertsCountWidget()::
    widgets.timeseries(
      title='Number of reverts in the last 7 days',
      query='avg:commits.reverts{source:softmax-system-health}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Should have 0 reverts in the last 7 days ',
            value: 'y = 0',
            display_type: 'warning dashed',
          },
        ],
      }
    ),

  // Hotfix count
  hotfixCountWidget()::
    widgets.timeseries(
      title='Number of hotfixes in the last 7 days',
      query='avg:commits.hotfix{source:softmax-system-health}',
      options={
        line_width: 'thick',
        markers: [
          {
            label: ' Should have 0 hotfixes in the last 7 days ',
            value: 'y = 0',
            display_type: 'warning dashed',
          },
        ],
      }
    ),
}

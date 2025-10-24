// EC2 infrastructure monitoring components
// Reusable widgets for EC2 instance, EBS, and cost metrics

local widgets = import '../lib/widgets.libsonnet';

{
  // Instance count widgets
  totalInstancesWidget():: widgets.queryValue(
    title='Total EC2 Instances',
    query='avg:ec2.instances.total{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  runningInstancesWidget():: widgets.queryValue(
    title='Running Instances',
    query='avg:ec2.instances.running{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  stoppedInstancesWidget():: widgets.queryValue(
    title='Stopped Instances',
    query='avg:ec2.instances.stopped{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  spotInstancesWidget():: widgets.queryValue(
    title='Spot Instances',
    query='avg:ec2.instances.spot{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  onDemandInstancesWidget():: widgets.queryValue(
    title='On-Demand Instances',
    query='avg:ec2.instances.ondemand{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  // Resource widgets
  gpuCountWidget():: widgets.queryValue(
    title='GPU Instances',
    query='avg:ec2.instances.gpu_count{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  cpuCountWidget():: widgets.queryValue(
    title='Total vCPUs',
    query='avg:ec2.instances.cpu_count{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: 'vCPUs',
    }
  ),

  idleInstancesWidget():: widgets.queryValue(
    title='Idle Instances',
    query='avg:ec2.instances.idle{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  // Age widgets
  avgInstanceAgeWidget():: widgets.queryValue(
    title='Average Instance Age',
    query='avg:ec2.instances.avg_age_days{*}',
    options={
      precision: 1,
      aggregator: 'last',
      custom_unit: 'days',
    }
  ),

  oldestInstanceWidget():: widgets.queryValue(
    title='Oldest Instance',
    query='avg:ec2.instances.oldest_age_days{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: 'days',
    }
  ),

  // EBS widgets
  totalVolumesWidget():: widgets.queryValue(
    title='Total EBS Volumes',
    query='avg:ec2.ebs.volumes.total{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  attachedVolumesWidget():: widgets.queryValue(
    title='Attached Volumes',
    query='avg:ec2.ebs.volumes.attached{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  unattachedVolumesWidget():: widgets.queryValue(
    title='Unattached Volumes',
    query='avg:ec2.ebs.volumes.unattached{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  totalStorageWidget():: widgets.queryValue(
    title='Total EBS Storage',
    query='avg:ec2.ebs.volumes.size_gb{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: 'GB',
    }
  ),

  snapshotsWidget():: widgets.queryValue(
    title='EBS Snapshots',
    query='avg:ec2.ebs.snapshots.total{*}',
    options={
      precision: 0,
      aggregator: 'last',
    }
  ),

  snapshotStorageWidget():: widgets.queryValue(
    title='Snapshot Storage',
    query='avg:ec2.ebs.snapshots.size_gb{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: 'GB',
    }
  ),

  // Cost widgets
  hourlyCostWidget():: widgets.queryValue(
    title='Hourly Cost Estimate',
    query='avg:ec2.cost.running_hourly_estimate{*}',
    options={
      precision: 2,
      aggregator: 'last',
      custom_unit: '$/hr',
    }
  ),

  monthlyCostWidget():: widgets.queryValue(
    title='Monthly Cost Estimate',
    query='avg:ec2.cost.monthly_estimate{*}',
    options={
      precision: 0,
      aggregator: 'last',
      custom_unit: '$/month',
    }
  ),

  spotSavingsWidget():: widgets.queryValue(
    title='Spot Savings %',
    query='avg:ec2.cost.spot_savings_pct{*}',
    options={
      precision: 1,
      aggregator: 'last',
      custom_unit: '%',
    }
  ),

  // Timeseries widgets
  instanceCountTrendWidget():: widgets.timeseries(
    title='EC2 Instance Count Trend',
    query='avg:ec2.instances.running{*}',
    options={
      show_legend: true,
      palette: 'blue',
    }
  ),

  costTrendWidget():: widgets.timeseries(
    title='Cost Trend (Hourly)',
    query='avg:ec2.cost.running_hourly_estimate{*}',
    options={
      show_legend: true,
      palette: 'purple',
      display_type: 'area',
    }
  ),

  spotVsOnDemandWidget():: widgets.timeseries(
    title='Spot vs On-Demand Instances',
    query='avg:ec2.instances.spot{*}',
    options={
      show_legend: true,
      palette: 'green',
      display_type: 'bars',
    }
  ),

  storageTrendWidget():: widgets.timeseries(
    title='EBS Storage Trend',
    query='avg:ec2.ebs.volumes.size_gb{*}',
    options={
      show_legend: true,
      palette: 'orange',
    }
  ),
}

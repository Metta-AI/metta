// EC2 Infrastructure Dashboard
// AWS EC2 instances, EBS volumes, and cost monitoring

local ec2 = import '../components/ec2.libsonnet';
local widgets = import '../lib/widgets.libsonnet';

{
  title: 'EC2 Infrastructure',
  description: 'AWS EC2 instance monitoring, EBS storage, and cost tracking across all regions',
  layout_type: 'ordered',
  template_variables: [],
  widgets: [
    // Header
    widgets.note(
      '## ☁️ EC2 Infrastructure Dashboard\n\n' +
      'Comprehensive monitoring of AWS EC2 instances, storage, and costs.\n\n' +
      '**Metrics Categories:**\n' +
      '- **Instance Metrics**: Total, running, stopped, spot vs on-demand, GPU/CPU counts\n' +
      '- **EBS Storage**: Volumes, snapshots, attached/unattached, storage sizes\n' +
      '- **Cost Tracking**: Hourly/monthly estimates, spot savings percentage\n\n' +
      '_Updated every 5 minutes via automated collector_',
      options={
        background_color: 'blue',
        font_size: '14',
        text_align: 'left',
        has_padding: true,
      }
    ),

    // Section 1: Instance Counts
    widgets.group(
      title='💻 EC2 Instances',
      widgets=[
        ec2.totalInstancesWidget(),
        ec2.runningInstancesWidget(),
        ec2.stoppedInstancesWidget(),
        ec2.idleInstancesWidget(),
        ec2.instanceCountTrendWidget(),
      ],
      options={
        background_color: 'vivid_blue',
        layout_type: 'ordered',
      }
    ),

    // Section 2: Instance Types & Resources
    widgets.group(
      title='🔧 Resources & Instance Types',
      widgets=[
        ec2.spotInstancesWidget(),
        ec2.onDemandInstancesWidget(),
        ec2.gpuCountWidget(),
        ec2.cpuCountWidget(),
        ec2.spotVsOnDemandWidget(),
      ],
      options={
        background_color: 'vivid_green',
        layout_type: 'ordered',
      }
    ),

    // Section 3: Instance Age
    widgets.group(
      title='📅 Instance Age',
      widgets=[
        ec2.avgInstanceAgeWidget(),
        ec2.oldestInstanceWidget(),
        widgets.note(
          '### Instance Age Analysis\n\n' +
          'Track instance lifecycle to identify long-running instances that may need review.\n\n' +
          '**Guidelines:**\n' +
          '- Average age < 30 days: Normal churn\n' +
          '- Average age > 90 days: May indicate persistent infrastructure\n' +
          '- Oldest > 180 days: Review for cleanup opportunities\n\n' +
          'Long-running instances may be candidates for Reserved Instances or Savings Plans.',
          options={
            background_color: 'gray',
            font_size: '12',
            text_align: 'left',
          }
        ),
      ],
      options={
        background_color: 'vivid_yellow',
        layout_type: 'ordered',
      }
    ),

    // Section 4: EBS Storage
    widgets.group(
      title='💾 EBS Storage',
      widgets=[
        ec2.totalVolumesWidget(),
        ec2.attachedVolumesWidget(),
        ec2.unattachedVolumesWidget(),
        ec2.totalStorageWidget(),
        ec2.snapshotsWidget(),
        ec2.snapshotStorageWidget(),
        ec2.storageTrendWidget(),
      ],
      options={
        background_color: 'vivid_orange',
        layout_type: 'ordered',
      }
    ),

    // Section 5: Cost Tracking
    widgets.group(
      title='💰 Cost Tracking',
      widgets=[
        ec2.hourlyCostWidget(),
        ec2.monthlyCostWidget(),
        ec2.spotSavingsWidget(),
        ec2.costTrendWidget(),
        widgets.note(
          '### Cost Optimization\n\n' +
          '**Current Status:**\n' +
          '- Hourly estimate based on simplified pricing model\n' +
          '- Spot savings shows percentage of running costs saved by using spot instances\n\n' +
          '**Optimization Opportunities:**\n' +
          '- Increase spot instance usage for cost savings\n' +
          '- Review unattached volumes for cleanup\n' +
          '- Consider Reserved Instances for long-running workloads\n\n' +
          '_Note: Cost estimates are approximate. Use AWS Cost Explorer for precise billing._',
          options={
            background_color: 'gray',
            font_size: '12',
            text_align: 'left',
          }
        ),
      ],
      options={
        background_color: 'vivid_purple',
        layout_type: 'ordered',
      }
    ),

    // Footer
    widgets.note(
      '---\n\n' +
      '**Total Metrics**: 19 metrics across 5 sections\n\n' +
      '**Data Source**: AWS EC2 API via automated collector\n' +
      '**Collection Frequency**: Every 5 minutes\n' +
      '**Component Library**: `components/ec2.libsonnet`\n\n' +
      '_Dashboard source: `dashboards/ec2.jsonnet`_',
      options={
        background_color: 'gray',
        font_size: '12',
        text_align: 'left',
      }
    ),
  ],
}

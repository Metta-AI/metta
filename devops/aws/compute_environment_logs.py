#!/usr/bin/env python3
"""
AWS Batch Compute Environment Logs

This script fetches CloudTrail logs related to AWS Batch compute environments
to help diagnose scaling issues, quota limits, and other problems.

Example usage:
    # Basic usage - get logs for a compute environment (only RunInstances events)
    python compute_environment_logs.py --compute-env metta-ce-multi-node

    # Focus on quota errors only
    python compute_environment_logs.py --compute-env metta-ce-multi-node --focus quota

    # Specify AWS profile
    python compute_environment_logs.py --compute-env metta-ce-multi-node --profile stem

    # Limit events to reduce runtime
    python compute_environment_logs.py --compute-env metta-ce-multi-node --max-events 50

    # Include all event types (slower)
    python compute_environment_logs.py --compute-env metta-ce-multi-node --event-types all
"""

import argparse
import boto3
import json
import sys
import os
from datetime import datetime, timedelta
from botocore.config import Config
from tabulate import tabulate

# Configure boto3 to use a higher max_pool_connections
config = Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)

def get_compute_environment_details(compute_env_name):
    """
    Get details about the specified compute environment.
    """
    batch = boto3.client('batch', config=config)

    try:
        response = batch.describe_compute_environments(
            computeEnvironments=[compute_env_name]
        )

        if not response['computeEnvironments']:
            print(f"Error: Compute environment '{compute_env_name}' not found.")
            sys.exit(1)

        return response['computeEnvironments'][0]
    except Exception as e:
        print(f"Error retrieving compute environment details: {str(e)}")
        sys.exit(1)

def get_ecs_cluster_from_compute_env(compute_env_details):
    """
    Extract the ECS cluster ARN from compute environment details.
    """
    return compute_env_details.get('ecsClusterArn')

def get_auto_scaling_groups(compute_env_name):
    """
    Get Auto Scaling groups associated with the compute environment.
    """
    autoscaling = boto3.client('autoscaling', config=config)

    try:
        response = autoscaling.describe_auto_scaling_groups()
        asg_list = []

        for asg in response['AutoScalingGroups']:
            # Check if this ASG is associated with our compute environment
            for tag in asg.get('Tags', []):
                if (tag['Key'] == 'AWSBatchComputeEnvironment' and
                    compute_env_name in tag['Value']):
                    asg_list.append(asg)
                    break

        return asg_list
    except Exception as e:
        print(f"Error retrieving Auto Scaling groups: {str(e)}")
        return []

def get_cloudtrail_events(compute_env_name, event_names, hours=24, max_events_per_type=100):
    """
    Get CloudTrail events related to the compute environment.

    Args:
        compute_env_name: Name of the compute environment
        event_names: List of event names to search for
        hours: Number of hours to look back
        max_events_per_type: Maximum number of events to fetch per event type
    """
    cloudtrail = boto3.client('cloudtrail', config=Config(
        retries={'max_attempts': 3, 'mode': 'standard'},
        read_timeout=30,
        connect_timeout=30,
        max_pool_connections=10
    ))

    # Calculate start time
    start_time = datetime.now() - timedelta(hours=hours)

    all_events = []

    try:
        # Process each event name separately
        for event_name in event_names:
            print(f"  Fetching {event_name} events...", end='', flush=True)
            next_token = None
            event_count = 0
            page_count = 0
            max_pages = 10  # Limit the number of pages to prevent hanging

            try:
                while page_count < max_pages and event_count < max_events_per_type:
                    page_count += 1

                    # Set up the lookup parameters
                    kwargs = {
                        'StartTime': start_time,
                        'LookupAttributes': [
                            {
                                'AttributeKey': 'EventName',
                                'AttributeValue': event_name
                            }
                        ],
                        'MaxResults': 50  # AWS maximum is 50
                    }

                    if next_token:
                        kwargs['NextToken'] = next_token

                    # Add timeout to prevent hanging
                    try:
                        # Use a timeout to prevent hanging
                        response = cloudtrail.lookup_events(**kwargs)
                    except Exception as e:
                        print(f" ERROR: API call failed: {str(e)}")
                        break

                    # Process the events from this page
                    relevant_events_count = 0
                    for event in response['Events']:
                        try:
                            cloud_trail_event = json.loads(event['CloudTrailEvent'])

                            # Check if this event is related to our compute environment
                            if is_event_related_to_compute_env(cloud_trail_event, compute_env_name):
                                all_events.append({
                                    'EventTime': event['EventTime'],
                                    'EventName': event['EventName'],
                                    'Username': event.get('Username', 'N/A'),
                                    'ErrorCode': cloud_trail_event.get('errorCode', ''),
                                    'ErrorMessage': cloud_trail_event.get('errorMessage', ''),
                                    'RequestParameters': cloud_trail_event.get('requestParameters', {}),
                                    'ResponseElements': cloud_trail_event.get('responseElements', {})
                                })
                                event_count += 1
                                relevant_events_count += 1

                                # Break early if we've reached the maximum events
                                if event_count >= max_events_per_type:
                                    break
                        except json.JSONDecodeError:
                            # Skip events with invalid JSON
                            continue
                        except Exception as e:
                            # Skip events that cause other errors
                            print(f" ERROR processing event: {str(e)}")
                            continue

                    # Print progress indicator
                    if page_count % 2 == 0:
                        print(".", end='', flush=True)

                    # Check if there are more events to retrieve
                    next_token = response.get('NextToken')
                    if not next_token or relevant_events_count == 0:
                        # If we got no relevant events on this page, don't continue pagination
                        # This helps prevent unnecessary API calls
                        break

                print(f" found {event_count} relevant events (searched {page_count} pages)")

                # If we hit the page limit, show a warning
                if page_count >= max_pages and next_token:
                    print(f"  WARNING: Reached maximum page limit ({max_pages}). There may be more events.")

            except Exception as e:
                print(f" ERROR: {str(e)}")
                continue

        # Sort events by time
        all_events.sort(key=lambda x: x['EventTime'], reverse=True)

        return all_events
    except Exception as e:
        print(f"Error retrieving CloudTrail events: {str(e)}")
        return []

def is_event_related_to_compute_env(event, compute_env_name):
    """
    Check if a CloudTrail event is related to the specified compute environment.
    """
    # Check request parameters
    request_params = event.get('requestParameters', {})
    response_elements = event.get('responseElements', {})

    # For RunInstances events, check if it's related to our compute environment
    if event.get('eventName') == 'RunInstances':
        # Check tags in the request
        tag_specs = request_params.get('tagSpecificationSet', {}).get('items', [])
        for tag_spec in tag_specs:
            tags = tag_spec.get('tags', [])
            for tag in tags:
                # Check for AWSBatchServiceTag and compute environment name in tags
                if tag.get('key') == 'AWSBatchServiceTag' and tag.get('value') == 'batch':
                    # For Batch service tags, we need to check if there's a compute environment tag
                    for tag2 in tags:
                        if tag2.get('key') == 'aws:batch:compute-environment':
                            # Check if this tag value exactly matches our compute environment name
                            if tag2.get('value') == compute_env_name:
                                return True
                            # If the tag contains the ARN, extract the name and compare
                            elif 'compute-environment/' + compute_env_name in tag2.get('value', ''):
                                return True
                    # If we didn't find a matching compute environment tag, return False
                    return False

                # Direct match on compute environment name in any tag value
                if compute_env_name == tag.get('value'):
                    return True

                # Check for compute environment in tag value (for ARN format)
                if tag.get('key') == 'aws:batch:compute-environment':
                    if tag.get('value') == compute_env_name or 'compute-environment/' + compute_env_name in tag.get('value', ''):
                        return True

        # If we've checked all tags and didn't find a match, this RunInstances is not for our compute env
        return False

    # For CreateComputeEnvironment or UpdateComputeEnvironment
    elif event.get('eventName') in ['CreateComputeEnvironment', 'UpdateComputeEnvironment']:
        compute_env = request_params.get('computeEnvironmentName', '')
        return compute_env_name == compute_env

    # For DeleteComputeEnvironment
    elif event.get('eventName') == 'DeleteComputeEnvironment':
        compute_env = request_params.get('computeEnvironment', '')
        # Check for exact match or ARN format
        return compute_env_name == compute_env or compute_env.endswith('/' + compute_env_name)

    # For TerminateInstances events, we need to check if the instance was part of our compute env
    elif event.get('eventName') == 'TerminateInstances':
        # Check if the instance IDs are in the response elements
        instance_ids = request_params.get('instancesSet', {}).get('items', [])
        if not instance_ids:
            return False

        # We can only include this if there's a clear reference to our compute environment
        # in the event data, otherwise we'll get too many false positives
        request_params_str = json.dumps(request_params)
        response_elements_str = json.dumps(response_elements)

        return (compute_env_name in request_params_str or
                compute_env_name in response_elements_str)

    # For other events, be more strict about matching
    else:
        # For other events, only include if there's an exact match on the compute environment name
        # in a key field, not just anywhere in the JSON

        # Check for compute environment name in specific fields
        if request_params.get('computeEnvironment') == compute_env_name:
            return True
        if request_params.get('computeEnvironmentName') == compute_env_name:
            return True

        # Check for ARN format
        if isinstance(request_params.get('computeEnvironment'), str) and request_params.get('computeEnvironment').endswith('/' + compute_env_name):
            return True

        # Check in job queue definitions (which reference compute environments)
        if 'computeEnvironmentOrder' in request_params:
            for env_order in request_params.get('computeEnvironmentOrder', []):
                if env_order.get('computeEnvironment') == compute_env_name:
                    return True
                if isinstance(env_order.get('computeEnvironment'), str) and env_order.get('computeEnvironment').endswith('/' + compute_env_name):
                    return True

        # Don't do a general string search as it's too prone to false positives
        return False

def get_service_quota_usage(service_code, quota_code):
    """
    Get the current usage for a specific service quota.
    """
    client = boto3.client('service-quotas', config=config)

    try:
        response = client.get_service_quota(
            ServiceCode=service_code,
            QuotaCode=quota_code
        )

        quota = response['Quota']

        # Get the CloudWatch metric for this quota
        if 'UsageMetric' in quota:
            metric = quota['UsageMetric']

            # Get the current usage from CloudWatch
            cloudwatch = boto3.client('cloudwatch', config=config)

            response = cloudwatch.get_metric_statistics(
                Namespace=metric['MetricNamespace'],
                MetricName=metric['MetricName'],
                Dimensions=[
                    {'Name': key, 'Value': value}
                    for key, value in metric['MetricDimensions'].items()
                ],
                StartTime=datetime.now() - timedelta(hours=3),
                EndTime=datetime.now(),
                Period=300,  # 5-minute periods
                Statistics=['Maximum']
            )

            # Get the most recent datapoint
            if response['Datapoints']:
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)
                current_usage = datapoints[0]['Maximum']

                return {
                    'QuotaName': quota['QuotaName'],
                    'QuotaValue': quota['Value'],
                    'CurrentUsage': current_usage,
                    'Unit': quota['Unit']
                }

        return {
            'QuotaName': quota['QuotaName'],
            'QuotaValue': quota['Value'],
            'CurrentUsage': 'Unknown',
            'Unit': quota['Unit']
        }
    except Exception as e:
        print(f"Error retrieving service quota usage: {str(e)}")
        return None

def get_ec2_instances(instance_lifecycle=None):
    """
    Get information about running EC2 instances.

    Args:
        instance_lifecycle: Optional filter for instance lifecycle (spot or on-demand)
    """
    ec2 = boto3.client('ec2', config=config)

    filters = [
        {
            'Name': 'instance-state-name',
            'Values': ['pending', 'running']
        }
    ]

    # Add instance lifecycle filter if specified
    if instance_lifecycle:
        filters.append({
            'Name': 'instance-lifecycle',
            'Values': [instance_lifecycle]
        })

    try:
        response = ec2.describe_instances(Filters=filters)

        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                # Determine if this is a spot or on-demand instance
                lifecycle = instance.get('InstanceLifecycle', 'on-demand')

                # Get instance tags
                tags = {}
                if 'Tags' in instance:
                    for tag in instance['Tags']:
                        tags[tag['Key']] = tag['Value']

                instances.append({
                    'InstanceId': instance['InstanceId'],
                    'InstanceType': instance['InstanceType'],
                    'State': instance['State']['Name'],
                    'LaunchTime': instance['LaunchTime'],
                    'Lifecycle': lifecycle,
                    'Tags': tags
                })

        return instances
    except Exception as e:
        print(f"Error retrieving EC2 instances: {str(e)}")
        return []

def get_ec2_spot_instances():
    """
    Get information about running spot instances.
    """
    return get_ec2_instances(instance_lifecycle='spot')

def get_ec2_ondemand_instances():
    """
    Get information about running on-demand instances.
    """
    # On-demand instances don't have an instance-lifecycle tag
    # We'll get all instances and filter out spot instances
    all_instances = get_ec2_instances()
    return [instance for instance in all_instances if instance['Lifecycle'] == 'on-demand']

def print_compute_env_details(compute_env_details):
    """
    Print details about the compute environment.
    """
    print("\n=== Compute Environment Details ===")
    print(f"Name: {compute_env_details['computeEnvironmentName']}")
    print(f"ARN: {compute_env_details['computeEnvironmentArn']}")
    print(f"State: {compute_env_details['state']}")
    print(f"Status: {compute_env_details['status']}")

    if 'statusReason' in compute_env_details:
        print(f"Status Reason: {compute_env_details['statusReason']}")

    if 'computeResources' in compute_env_details:
        compute_resources = compute_env_details['computeResources']
        print("\nCompute Resources:")
        print(f"  Type: {compute_resources.get('type', 'N/A')}")
        print(f"  Min vCPUs: {compute_resources.get('minvCpus', 'N/A')}")
        print(f"  Max vCPUs: {compute_resources.get('maxvCpus', 'N/A')}")
        print(f"  Desired vCPUs: {compute_resources.get('desiredvCpus', 'N/A')}")

        if 'instanceTypes' in compute_resources:
            print(f"  Instance Types: {', '.join(compute_resources['instanceTypes'])}")

        if 'subnets' in compute_resources:
            print(f"  Subnets: {', '.join(compute_resources['subnets'])}")

        if 'securityGroupIds' in compute_resources:
            print(f"  Security Groups: {', '.join(compute_resources['securityGroupIds'])}")

def print_auto_scaling_groups(asg_list):
    """
    Print details about Auto Scaling groups.
    """
    if not asg_list:
        print("\n=== No Auto Scaling Groups Found ===")
        return

    print("\n=== Auto Scaling Groups ===")
    for asg in asg_list:
        print(f"Name: {asg['AutoScalingGroupName']}")
        print(f"Min Size: {asg['MinSize']}")
        print(f"Max Size: {asg['MaxSize']}")
        print(f"Desired Capacity: {asg['DesiredCapacity']}")
        print(f"Instances: {len(asg['Instances'])}")

        if asg['Instances']:
            print("\nInstances:")
            for instance in asg['Instances']:
                print(f"  {instance['InstanceId']} - {instance['LifecycleState']}")

        print("\n")

def print_instances(instances, instance_type="EC2"):
    """
    Print information about EC2 instances.
    """
    if not instances:
        print(f"\n=== No {instance_type} Instances Found ===")
        return

    print(f"\n=== {instance_type} Instances ===")

    # Prepare table data
    table_data = []
    for instance in instances:
        # Format launch time
        launch_time = instance['LaunchTime']
        if isinstance(launch_time, datetime):
            launch_time = launch_time.strftime('%Y-%m-%d %H:%M:%S')

        # Get compute environment tag if available
        compute_env = instance['Tags'].get('aws:batch:compute-environment', 'N/A')

        table_data.append([
            instance['InstanceId'],
            instance['InstanceType'],
            instance['State'],
            launch_time,
            compute_env
        ])

    # Print table
    print(tabulate(table_data, headers=['Instance ID', 'Type', 'State', 'Launch Time', 'Compute Environment'], tablefmt='grid'))

def print_spot_instances(instances):
    """
    Print information about spot instances.
    """
    print_instances(instances, instance_type="Spot")

def print_ondemand_instances(instances):
    """
    Print information about on-demand instances.
    """
    print_instances(instances, instance_type="On-Demand")

def print_cloudtrail_events(events, max_events=10):
    """
    Print CloudTrail events in a readable format.

    Args:
        events: List of CloudTrail events
        max_events: Maximum number of events to display
    """
    if not events:
        print("\n=== No CloudTrail Events Found ===")
        return

    # Limit to max_events most recent events
    events_to_display = events[:max_events]

    print(f"\n=== CloudTrail Events (Showing {len(events_to_display)} most recent of {len(events)} events) ===")

    # Prepare table data
    table_data = []
    for event in events_to_display:
        error = f"{event['ErrorCode']}: {event['ErrorMessage']}" if event['ErrorCode'] else ""

        # Format event time
        event_time = event['EventTime']
        if isinstance(event_time, datetime):
            event_time = event_time.strftime('%Y-%m-%d %H:%M:%S')

        table_data.append([
            event_time,
            event['EventName'],
            event['Username'],
            error[:100] + ('...' if len(error) > 100 else '')  # Truncate long error messages
        ])

    # Print table
    print(tabulate(table_data, headers=['Time', 'Event', 'User', 'Error'], tablefmt='grid'))

    # Print detailed information for events with errors
    print("\n=== Detailed Error Information ===")
    error_found = False

    for event in events_to_display:
        if event['ErrorCode']:
            error_found = True
            print(f"\nEvent: {event['EventName']} at {event['EventTime']}")
            print(f"Error: {event['ErrorCode']}: {event['ErrorMessage']}")

            # Print relevant request parameters
            if event['RequestParameters']:
                print("\nRequest Parameters:")
                if event['EventName'] == 'RunInstances':
                    instance_type = event['RequestParameters'].get('instanceType', 'N/A')
                    subnet_id = event['RequestParameters'].get('subnetId', 'N/A')

                    print(f"  Instance Type: {instance_type}")
                    print(f"  Subnet ID: {subnet_id}")

                    # Print market options if available
                    market_options = event['RequestParameters'].get('instanceMarketOptions', {})
                    if market_options:
                        market_type = market_options.get('marketType', 'N/A')
                        print(f"  Market Type: {market_type}")

                elif event['EventName'] in ['CreateAutoScalingGroup', 'UpdateAutoScalingGroup']:
                    min_size = event['RequestParameters'].get('MinSize', 'N/A')
                    max_size = event['RequestParameters'].get('MaxSize', 'N/A')
                    desired_capacity = event['RequestParameters'].get('DesiredCapacity', 'N/A')

                    print(f"  Min Size: {min_size}")
                    print(f"  Max Size: {max_size}")
                    print(f"  Desired Capacity: {desired_capacity}")

    # Print summary of quota-related errors
    quota_errors = [e for e in events_to_display if 'quota' in e['ErrorMessage'].lower() or
                                         'limit' in e['ErrorMessage'].lower() or
                                         'exceeded' in e['ErrorMessage'].lower() or
                                         'MaxSpotInstanceCountExceeded' in e['ErrorCode'] or
                                         'SpotInstanceCountExceeded' in e['ErrorCode'] or
                                         'VcpuLimitExceeded' in e['ErrorCode'] or
                                         'InstanceLimitExceeded' in e['ErrorCode']]

    if quota_errors:
        print("\n=== Quota/Limit Related Errors ===")
        for error in quota_errors:
            print(f"\n{error['EventTime']} - {error['EventName']}")
            print(f"Error: {error['ErrorCode']}: {error['ErrorMessage']}")

            # For RunInstances errors, show the instance type
            if error['EventName'] == 'RunInstances' and 'instanceType' in error['RequestParameters']:
                print(f"Instance Type: {error['RequestParameters']['instanceType']}")

    # Print summary of instance launch failures
    launch_failures = [e for e in events_to_display if e['EventName'] == 'RunInstances' and e['ErrorCode']]

    if launch_failures:
        print("\n=== Instance Launch Failures ===")
        failure_counts = {}

        for failure in launch_failures:
            error_key = f"{failure['ErrorCode']}: {failure['ErrorMessage']}"
            if error_key in failure_counts:
                failure_counts[error_key] += 1
            else:
                failure_counts[error_key] = 1

        for error, count in failure_counts.items():
            print(f"{count} failures: {error}")

    if not error_found and not quota_errors and not launch_failures:
        print("No errors found in the CloudTrail events.")

def print_service_quotas(quotas):
    """
    Print service quota information.
    """
    if not quotas:
        print("\n=== No Service Quota Information Available ===")
        return

    print("\n=== Service Quotas ===")

    # Prepare table data
    table_data = []
    for quota in quotas:
        if quota:
            usage_percent = "Unknown"
            if quota['CurrentUsage'] != 'Unknown' and quota['QuotaValue'] > 0:
                usage_percent = f"{(quota['CurrentUsage'] / quota['QuotaValue']) * 100:.1f}%"

            table_data.append([
                quota['QuotaName'],
                quota['QuotaValue'],
                quota['CurrentUsage'],
                usage_percent,
                quota['Unit']
            ])

    # Print table
    print(tabulate(table_data, headers=['Quota Name', 'Limit', 'Current Usage', 'Usage %', 'Unit'], tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description='Get logs and information about an AWS Batch compute environment.')
    parser.add_argument('--compute-env', required=True, help='The name of the compute environment')
    parser.add_argument('--hours', type=int, default=24, help='Number of hours of logs to retrieve (default: 24)')
    parser.add_argument('--profile', help='AWS profile to use')
    parser.add_argument('--event-types', help='Comma-separated list of event types to search for, or "all" for all events')
    parser.add_argument('--max-events', type=int, default=100, help='Maximum number of events to fetch per event type (default: 100)')
    parser.add_argument('--max-display', type=int, default=10, help='Maximum number of events to display (default: 10)')
    parser.add_argument('--focus', choices=['quota', 'launch', 'all'], default='launch',
                        help='Focus on specific error types: quota (quota/limit errors), launch (instance launch errors, default), all')
    args = parser.parse_args()

    # Set AWS profile if specified
    if args.profile:
        boto3.setup_default_session(profile_name=args.profile)

    # Get compute environment details
    compute_env_details = get_compute_environment_details(args.compute_env)
    print_compute_env_details(compute_env_details)

    # Define event names to search for based on focus
    if args.event_types == "all":
        # All relevant events
        event_names = [
            'RunInstances',
            'TerminateInstances',
            'CreateComputeEnvironment',
            'UpdateComputeEnvironment',
            'DeleteComputeEnvironment',
            'SubmitJob',
            'RegisterJobDefinition',
            'CreateJobQueue',
            'UpdateJobQueue',
            'CreateLaunchTemplate',
            'CreateFleet',
            'RequestSpotInstances',
            'CancelSpotInstanceRequests',
            'CreateCapacityProvider',
            'UpdateCapacityProvider'
        ]
    elif args.event_types:
        event_names = args.event_types.split(',')
    elif args.focus == 'quota':
        # Focus on events that might have quota errors
        event_names = ['RunInstances']
    elif args.focus == 'launch':
        # Focus on instance launch events
        event_names = ['RunInstances']
    else:
        # Default - just RunInstances events
        event_names = ['RunInstances']

    # Get CloudTrail events
    print(f"\nFetching CloudTrail events for the last {args.hours} hours...")
    events = get_cloudtrail_events(args.compute_env, event_names, args.hours, args.max_events)
    print_cloudtrail_events(events, args.max_display)

    # Get EC2 instances
    if args.focus == 'launch' or args.focus == 'all':
        # Get spot instances
        spot_instances = get_ec2_spot_instances()
        print_spot_instances(spot_instances)

        # Get on-demand instances
        ondemand_instances = get_ec2_ondemand_instances()
        print_ondemand_instances(ondemand_instances)

    # Get service quota information
    print("\nFetching service quota information...")
    quotas = [
        get_service_quota_usage('ec2', 'L-3819A6DF'),  # All G and VT Spot Instance Requests
        get_service_quota_usage('ec2', 'L-DB2E81BA')   # Running On-Demand G and VT instances
    ]
    print_service_quotas(quotas)

    # Print a summary of findings
    print("\n=== Summary ===")
    if args.focus == 'quota' or args.focus == 'all':
        quota_errors = [e for e in events if 'quota' in e['ErrorMessage'].lower() or
                                             'limit' in e['ErrorMessage'].lower() or
                                             'exceeded' in e['ErrorMessage'].lower() or
                                             'MaxSpotInstanceCountExceeded' in e['ErrorCode'] or
                                             'SpotInstanceCountExceeded' in e['ErrorCode'] or
                                             'VcpuLimitExceeded' in e['ErrorCode'] or
                                             'InstanceLimitExceeded' in e['ErrorCode']]
        if quota_errors:
            print(f"Found {len(quota_errors)} quota/limit related errors.")
            print("Recommendation: Consider requesting a quota increase or using different instance types.")

            # Check if it's specifically a spot instance quota issue
            spot_quota_errors = [e for e in quota_errors if 'spot' in e['ErrorMessage'].lower() or
                                                           'MaxSpotInstanceCountExceeded' in e['ErrorCode'] or
                                                           'SpotInstanceCountExceeded' in e['ErrorCode']]
            if spot_quota_errors:
                print("\nThis appears to be a Spot Instance quota issue.")
                print("To request a quota increase:")
                print("1. Go to AWS Service Quotas console: https://console.aws.amazon.com/servicequotas/")
                print("2. Select 'Amazon Elastic Compute Cloud (Amazon EC2)'")
                print("3. Search for 'All G and VT Spot Instance Requests'")
                print("4. Click 'Request quota increase'")
        else:
            print("No quota/limit related errors found.")

    if args.focus == 'launch' or args.focus == 'all':
        launch_failures = [e for e in events if e['EventName'] == 'RunInstances' and e['ErrorCode']]
        if launch_failures:
            print(f"Found {len(launch_failures)} instance launch failures.")
            print("Recommendation: Check the detailed error information above for specific issues.")
        else:
            print("No instance launch failures found.")

if __name__ == '__main__':
    main()

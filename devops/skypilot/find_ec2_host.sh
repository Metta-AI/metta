#!/bin/bash

set -e

JOB_ID="$1"

if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

aws ec2 describe-instances --filters "Name=tag:Name,Values=*-$JOB_ID-*" --query 'Reservations[0].Instances[0].PrivateDnsName' --output text | cat

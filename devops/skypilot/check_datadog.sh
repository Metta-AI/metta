#!/usr/bin/env bash
# Quick check if Datadog agent is working

JOB_ID=$1
RUN_ID=$2

if [ -z "$JOB_ID" ] || [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <job_id> <run_id>"
    exit 1
fi

echo "Job: $JOB_ID | Run: $RUN_ID"
echo ""
echo "Check Datadog Dashboard:"
echo "1. Infrastructure: https://app.datadoghq.com/infrastructure"
echo "   Search: metta_run_id:$RUN_ID"
echo ""
echo "2. Metrics: https://app.datadoghq.com/metric/explorer"
echo "   Query: system.cpu.user{metta_run_id:$RUN_ID}"
echo ""
echo "3. Logs: https://app.datadoghq.com/logs"
echo "   Filter: metta_run_id:$RUN_ID"
echo ""
echo "If you see data → Agent is working ✅"
echo "If empty → Agent not running ❌"


#!/bin/bash
# Cleanup script for dashboard-cronjob resources
# Run this to stop alerts from crashlooping pods

set -e

NAMESPACE="monitoring"

echo "🔍 Checking for dashboard-cronjob resources..."

# Suspend the main dashboard-cronjob to stop new pods
echo "⏸️  Suspending dashboard-cronjob-cronjob..."
kubectl -n "$NAMESPACE" patch cronjob dashboard-cronjob-cronjob -p '{"spec":{"suspend":true}}' 2>/dev/null && echo "✓ Suspended" || echo "⚠️  Not found (may already be removed)"

# Suspend the dev dashboard-cronjob if it exists
echo "⏸️  Suspending dashboard-cronjob-dev-cronjob..."
kubectl -n "$NAMESPACE" patch cronjob dashboard-cronjob-dev-cronjob -p '{"spec":{"suspend":true}}' 2>/dev/null && echo "✓ Suspended" || echo "⚠️  Not found (may not exist)"

# Delete specific crashlooping jobs mentioned in the alert
echo "🗑️  Deleting crashlooping jobs..."
kubectl -n "$NAMESPACE" delete job dashboard-cronjob-cronjob-29423400-kzwm8 2>/dev/null && echo "✓ Deleted dashboard-cronjob-cronjob-29423400-kzwm8" || echo "⚠️  Job not found"
kubectl -n "$NAMESPACE" delete job dashboard-cronjob-test-1765404523-52zhk 2>/dev/null && echo "✓ Deleted dashboard-cronjob-test-1765404523-52zhk" || echo "⚠️  Job not found"

# Delete all failed jobs from dashboard-cronjob
echo "🗑️  Cleaning up all failed dashboard-cronjob jobs..."
FAILED_JOBS=$(kubectl -n "$NAMESPACE" get jobs -l app.kubernetes.io/name=dashboard-cronjob --field-selector status.successful!=1 -o name 2>/dev/null || true)
if [ -n "$FAILED_JOBS" ]; then
    echo "$FAILED_JOBS" | xargs -r kubectl -n "$NAMESPACE" delete && echo "✓ Deleted failed jobs" || echo "⚠️  No failed jobs found"
else
    echo "✓ No failed jobs to clean up"
fi

# List remaining dashboard-cronjob resources
echo ""
echo "📋 Remaining dashboard-cronjob resources:"
kubectl -n "$NAMESPACE" get cronjobs,jobs,pods -l app.kubernetes.io/name=dashboard-cronjob 2>/dev/null || echo "No resources found"

echo ""
echo "✅ Cleanup complete! Alerts should stop once Datadog detects the pods are gone."

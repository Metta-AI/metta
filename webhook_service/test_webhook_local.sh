#!/bin/bash
set -e

echo "Testing GitHub webhook service locally..."

# Start the service in background
echo "Starting webhook service..."
cd "$(dirname "$0")"
uv run python -m github_webhook.app &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Test ping event
echo ""
echo "Testing ping event..."
curl -X POST http://localhost:8000/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: ping" \
  -H "X-GitHub-Delivery: test-delivery-123" \
  -d '{"zen": "test"}'

echo ""
echo ""

# Test pull_request opened event (without signature for now)
echo "Testing pull_request opened event..."
curl -X POST http://localhost:8000/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: pull_request" \
  -H "X-GitHub-Delivery: test-delivery-456" \
  -d '{
    "action": "opened",
    "pull_request": {
      "number": 9999,
      "title": "Test PR",
      "html_url": "https://github.com/Metta-AI/metta/pull/9999",
      "user": {"login": "testuser"},
      "assignee": null
    },
    "repository": {
      "full_name": "Metta-AI/metta"
    }
  }'

echo ""
echo ""

# Cleanup
echo "Stopping server..."
kill $SERVER_PID 2> /dev/null || true

echo "Test complete!"

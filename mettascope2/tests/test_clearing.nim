## Quick test to verify terrain clearing works
import mettascope2/src/mettascope/tribal
import std/strutils

let env = newEnvironment()
echo "Testing terrain clearing in houses/villages..."
echo ""
echo "Map (first 30 lines) - Look for clear areas inside house walls (#):"
echo ""

var lines = 0
for line in env.render().split('\n'):
  if lines < 30:
    echo line
    lines += 1

echo ""
echo "Houses should have clear interiors (no 't' or '.' inside walls)"
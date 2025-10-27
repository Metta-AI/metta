#!/bin/bash
while true; do
    clear
    echo "=== Difficulty Evaluation Progress ===" 
    echo "Time: $(date)"
    echo ""
    
    # Check if process is running
    if ps aux | grep -v grep | grep "evaluate_with_difficulties.py" > /dev/null; then
        echo "✅ Process running (PID: $(pgrep -f evaluate_with_difficulties))"
    else
        echo "❌ Process not running"
        break
    fi
    
    # Log size
    if [ -f difficulty_evaluation.log ]; then
        lines=$(wc -l < difficulty_evaluation.log)
        echo "📝 Log lines: $lines"
    fi
    
    # Latest results
    echo ""
    echo "=== Latest Results (last 10 tests) ==="
    grep -E "succeeded\]|Reward:|Hearts:|SUCCESS|FAILED" difficulty_evaluation.log | tail -20
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done

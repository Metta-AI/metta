#!/usr/bin/env python3
"""Query the stable tag history for analysis and reporting."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_history(history_file: Path) -> List[Dict]:
    """Load the stability history from JSON file."""
    if not history_file.exists():
        return []
    
    with open(history_file, 'r') as f:
        return json.load(f)


def format_date(date_str: str) -> str:
    """Format ISO date string to human readable format."""
    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d %H:%M UTC')


def get_latest_stable(history: List[Dict]) -> Optional[Dict]:
    """Get the latest stable release."""
    return history[-1] if history else None


def get_stats(history: List[Dict]) -> Dict:
    """Calculate statistics from the history."""
    if not history:
        return {
            'total_releases': 0,
            'avg_commits_between': 0,
            'avg_days_between': 0,
            'test_skip_rate': 0
        }
    
    total = len(history)
    commits_between = [e['commits_since_last'] for e in history if e['commits_since_last'] > 0]
    avg_commits = sum(commits_between) / len(commits_between) if commits_between else 0
    
    # Calculate average days between releases
    dates = [datetime.fromisoformat(e['date'].replace('Z', '+00:00')) for e in history]
    if len(dates) > 1:
        deltas = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_days = sum(deltas) / len(deltas)
    else:
        avg_days = 0
    
    # Calculate test skip rate
    skipped = sum(1 for e in history if e.get('tests_skipped', False))
    skip_rate = (skipped / total) * 100
    
    return {
        'total_releases': total,
        'avg_commits_between': avg_commits,
        'avg_days_between': avg_days,
        'test_skip_rate': skip_rate
    }


def print_history(history: List[Dict], limit: int = 10):
    """Print the recent history in a formatted table."""
    print("\nRecent Stable Releases:")
    print("-" * 80)
    print(f"{'Date':<20} {'Commit':<12} {'Commits Since':<15} {'Tests':<10}")
    print("-" * 80)
    
    for entry in history[-limit:]:
        date = format_date(entry['date'])
        commit = entry['commit'][:8]
        commits_since = entry['commits_since_last']
        tests = 'Skipped' if entry.get('tests_skipped', False) else 'Passed'
        
        print(f"{date:<20} {commit:<12} {commits_since:<15} {tests:<10}")


def main():
    parser = argparse.ArgumentParser(description='Query the stable tag history')
    parser.add_argument('--history-file', type=Path, 
                        default=Path('docs/stability/history.json'),
                        help='Path to the history JSON file')
    parser.add_argument('--latest', action='store_true',
                        help='Show only the latest stable release')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics about stable releases')
    parser.add_argument('--limit', type=int, default=10,
                        help='Number of recent releases to show (default: 10)')
    parser.add_argument('--json', action='store_true',
                        help='Output in JSON format')
    
    args = parser.parse_args()
    
    history = load_history(args.history_file)
    
    if not history:
        print("No stable release history found.", file=sys.stderr)
        sys.exit(1)
    
    if args.latest:
        latest = get_latest_stable(history)
        if args.json:
            print(json.dumps(latest, indent=2))
        else:
            print(f"\nLatest Stable Release:")
            print(f"  Commit: {latest['commit'][:8]}")
            print(f"  Date: {format_date(latest['date'])}")
            print(f"  Commits since previous: {latest['commits_since_last']}")
            print(f"  Tests: {'Skipped' if latest.get('tests_skipped', False) else 'Passed'}")
    
    elif args.stats:
        stats = get_stats(history)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("\nStability Statistics:")
            print(f"  Total releases: {stats['total_releases']}")
            print(f"  Average commits between releases: {stats['avg_commits_between']:.1f}")
            print(f"  Average days between releases: {stats['avg_days_between']:.1f}")
            print(f"  Test skip rate: {stats['test_skip_rate']:.1f}%")
    
    else:
        if args.json:
            print(json.dumps(history[-args.limit:], indent=2))
        else:
            print_history(history, args.limit)
            print("\nUse --stats to see statistics or --latest for the current stable release")


if __name__ == '__main__':
    main()
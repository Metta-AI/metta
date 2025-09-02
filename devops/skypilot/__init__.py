import sys
from devops.skypilot.src import skypilot_latency
from devops.skypilot.src.skypilot_latency import calculate_queue_latency, parse_submission_timestamp

# Make the module available as devops.skypilot.skypilot_latency
sys.modules['devops.skypilot.skypilot_latency'] = skypilot_latency

__all__ = [
    'calculate_queue_latency',
    'parse_submission_timestamp',
]

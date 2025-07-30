#!/usr/bin/env python3
"""Quick test to verify heart calculations"""


def test_heart_appearances(timing, max_steps=500):
    """Show when hearts appear for a given timing value"""
    period = 2 * timing  # This is the key relationship!

    print(f"\nTiming: {timing}")
    print(f"Period: {period}")
    print("Hearts appear at times: ", end="")

    heart_times = []
    time = period
    while time <= max_steps:
        heart_times.append(time)
        time += period

    print(heart_times)
    print(f"Total hearts in {max_steps} steps: {len(heart_times)}")
    print(f"Optimal collection rate: {max_steps / period:.2f} hearts")

    return len(heart_times)


# Test the timings mentioned
test_timings = [6, 8, 10, 11, 15, 16, 18, 20, 22, 25, 33, 50, 100]

print("Heart Appearance Analysis")
print("=" * 50)

for timing in test_timings:
    test_heart_appearances(timing)

# Special focus on timing 100
print("\n" + "=" * 50)
print("DETAILED LOOK AT TIMING 100:")
print("=" * 50)
timing_100_hearts = test_heart_appearances(100)
print(f"\nThis explains why timing 100 only yields {timing_100_hearts} hearts!")
print("The period is 200, not 100!")

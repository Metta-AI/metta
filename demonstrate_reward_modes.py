#!/usr/bin/env python3
"""Demonstrate the different ColorTree reward modes."""


def simulate_reward_modes():
    """Simulate different reward modes for ColorTree."""
    target_sequence = [0, 1, 2, 3]
    sequence_reward = 5.0

    print("ColorTree Reward Modes Demonstration")
    print("=" * 60)
    print(f"Target sequence: {target_sequence}")
    print(f"Total sequence reward: {sequence_reward}")
    print()

    # Test sequences
    test_sequences = [
        ([0, 1, 2, 3], "Perfect match"),
        ([0, 1, 2, 2], "3/4 correct"),
        ([0, 0, 2, 3], "3/4 correct (different positions)"),
        ([1, 2, 3, 0], "All wrong positions"),
        ([3, 2, 1, 0], "Complete reverse"),
    ]

    for sequence, description in test_sequences:
        print(f"\nSequence: {sequence} - {description}")
        print("-" * 40)

        # Precise mode
        precise_reward = 0
        if sequence == target_sequence:
            precise_reward = sequence_reward

        # Partial mode
        correct_positions = sum(1 for i in range(4) if sequence[i] == target_sequence[i])
        partial_reward = sequence_reward * (correct_positions / 4)

        # Dense mode (simulated step by step)
        dense_rewards = []
        for i, action in enumerate(sequence):
            if action == target_sequence[i]:
                dense_rewards.append(sequence_reward / 4)
            else:
                dense_rewards.append(0)
        dense_total = sum(dense_rewards)

        print(f"  Precise mode: {precise_reward:.2f} reward")
        print(f"  Partial mode: {partial_reward:.2f} reward ({correct_positions}/4 correct)")
        print(f"  Dense mode:   {dense_total:.2f} reward (per-action: {[f'{r:.2f}' for r in dense_rewards]})")

    print("\n" + "=" * 60)
    print("Summary:")
    print("- Precise: All-or-nothing reward at sequence completion")
    print("- Partial: Proportional reward based on correct positions")
    print("- Dense: Immediate reward for each correct action")
    print()
    print("Dense mode provides the most immediate feedback,")
    print("making it easier for agents to learn the sequence.")


if __name__ == "__main__":
    simulate_reward_modes()

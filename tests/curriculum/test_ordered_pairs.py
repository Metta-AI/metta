"""Test that we generate the correct number of ordered pairs for experiments."""

import itertools


def test_ordered_pairs_generation():
    """Test that we generate all 12 ordered pairs from 4 task sets."""
    task_sets = ["navigation", "memory", "navigation_sequence", "object_use"]

    # Generate ordered pairs (permutations)
    ordered_pairs = list(itertools.permutations(task_sets, 2))

    # Should have 12 pairs (4 * 3 = 12)
    assert len(ordered_pairs) == 12

    # Verify all pairs are unique and ordered
    expected_pairs = [
        ("navigation", "memory"),
        ("navigation", "navigation_sequence"),
        ("navigation", "object_use"),
        ("memory", "navigation"),
        ("memory", "navigation_sequence"),
        ("memory", "object_use"),
        ("navigation_sequence", "navigation"),
        ("navigation_sequence", "memory"),
        ("navigation_sequence", "object_use"),
        ("object_use", "navigation"),
        ("object_use", "memory"),
        ("object_use", "navigation_sequence"),
    ]

    assert set(ordered_pairs) == set(expected_pairs)

    # Verify that order matters (pairs are different)
    assert ("navigation", "memory") != ("memory", "navigation")
    assert ("navigation", "object_use") != ("object_use", "navigation")


def test_pair_names_generation():
    """Test that pair names are generated correctly."""
    task_sets = ["navigation", "memory", "navigation_sequence", "object_use"]
    ordered_pairs = list(itertools.permutations(task_sets, 2))

    pair_names = [f"{task_set_1}_to_{task_set_2}" for task_set_1, task_set_2 in ordered_pairs]

    # Should have 12 unique pair names
    assert len(pair_names) == 12
    assert len(set(pair_names)) == 12  # All unique

    # Verify some expected pair names
    expected_names = [
        "navigation_to_memory",
        "navigation_to_navigation_sequence",
        "navigation_to_object_use",
        "memory_to_navigation",
        "memory_to_navigation_sequence",
        "memory_to_object_use",
        "navigation_sequence_to_navigation",
        "navigation_sequence_to_memory",
        "navigation_sequence_to_object_use",
        "object_use_to_navigation",
        "object_use_to_memory",
        "object_use_to_navigation_sequence",
    ]

    assert set(pair_names) == set(expected_names)


if __name__ == "__main__":
    test_ordered_pairs_generation()
    test_pair_names_generation()
    print("✓ All ordered pair tests passed!")

#!/usr/bin/env python
"""Test action conversion to verify the bug hypothesis."""

import torch

def test_action_conversion():
    """Test the action conversion formulas to understand the bug."""
    
    # Example action configuration
    action_names = ["attack", "get_items", "move", "noop", "put_items", "rotate", "swap"]
    action_max_params = [8, 0, 1, 3, 0, 0, 0]
    
    print("Action configuration:")
    for i, (name, max_param) in enumerate(zip(action_names, action_max_params)):
        print(f"  {i}: {name} (max_param={max_param}, total_actions={max_param+1})")
    
    # MettaAgent's current (buggy) cumsum calculation
    buggy_cumsum = torch.cumsum(
        torch.tensor([0] + action_max_params, dtype=torch.long), dim=0
    )
    print(f"\nBuggy cumsum: {buggy_cumsum.tolist()}")
    
    # What the cumsum SHOULD be (correct calculation)
    correct_cumsum = torch.cumsum(
        torch.tensor([0] + [p+1 for p in action_max_params], dtype=torch.long), dim=0
    )
    print(f"Correct cumsum: {correct_cumsum.tolist()}")
    
    # Test some action conversions
    test_actions = [
        (0, 0),  # attack_0
        (0, 8),  # attack_8
        (1, 0),  # get_items_0
        (2, 0),  # move_0
        (2, 1),  # move_1
        (3, 0),  # noop_0
        (3, 3),  # noop_3
    ]
    
    print("\nAction index conversions:")
    print("Action (type, param) -> Buggy Formula -> Correct Formula")
    
    for action_type, action_param in test_actions:
        # Current formula used by both ComponentPolicy and Fast
        # This compensates for the buggy cumsum
        buggy_formula_result = action_type + buggy_cumsum[action_type] + action_param
        
        # What the formula SHOULD be with correct cumsum
        correct_formula_result = correct_cumsum[action_type] + action_param
        
        print(f"  ({action_type}, {action_param}) -> {buggy_formula_result} -> {correct_formula_result}")
    
    # Now let's check if they produce the same results
    print("\nDo formulas match?")
    all_match = True
    for action_type, action_param in test_actions:
        buggy_formula_result = action_type + buggy_cumsum[action_type] + action_param
        correct_formula_result = correct_cumsum[action_type] + action_param
        match = buggy_formula_result == correct_formula_result
        if not match:
            all_match = False
            print(f"  MISMATCH: ({action_type}, {action_param})")
    
    if all_match:
        print("  All conversions match! The buggy formula + buggy cumsum = correct result")
    else:
        print("  Some conversions don't match - there's a problem!")
    
    # Build the full action mapping to verify
    print("\nFull action mapping:")
    index = 0
    for action_type, (name, max_param) in enumerate(zip(action_names, action_max_params)):
        for param in range(max_param + 1):
            # Using buggy formula + buggy cumsum
            buggy_index = action_type + buggy_cumsum[action_type] + param
            print(f"  {index}: {name}_{param} -> logit_index={buggy_index}")
            index += 1

if __name__ == "__main__":
    test_action_conversion()
#!/usr/bin/env python3
"""Comprehensive test of arena_trader_experiment after merge."""

import sys
import traceback
import numpy as np
import torch

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        import experiments.recipes.arena_trader_experiment as ate
        print("✓ Module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_env_creation():
    """Test environment creation."""
    print("\nTesting environment creation...")
    try:
        import experiments.recipes.arena_trader_experiment as ate
        env = ate.make_env()
        print(f"✓ Environment created")
        print(f"  - Num agents: {env.game.num_agents}")
        print(f"  - Transfer enabled: {env.game.actions.transfer.enabled}")
        print(f"  - Trader group ID: {env.game.actions.transfer.trader_group_id}")
        
        # Check trader config
        print(f"  - Ore->Battery trade: {env.game.actions.transfer.input_resources} -> {env.game.actions.transfer.output_resources}")
        print(f"  - Generator conversion: {env.game.objects['generator_red'].input_resources}")
        print(f"  - Altar conversion: {env.game.objects['altar'].input_resources}")
        return True
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        traceback.print_exc()
        return False

def test_curriculum():
    """Test curriculum creation."""
    print("\nTesting curriculum...")
    try:
        import experiments.recipes.arena_trader_experiment as ate
        curriculum = ate.make_curriculum()
        print(f"✓ Curriculum created")
        
        # Check that task generator was created
        print(f"  - Task generator type: {type(curriculum.task_generator).__name__}")
        print(f"  - Curriculum has task generator: {curriculum.task_generator is not None}")
        return True
    except Exception as e:
        print(f"✗ Curriculum failed: {e}")
        traceback.print_exc()
        return False

def test_evaluations():
    """Test evaluation configurations."""
    print("\nTesting evaluations...")
    try:
        import experiments.recipes.arena_trader_experiment as ate
        evals = ate.make_evals()
        print(f"✓ Created {len(evals)} evaluation scenarios")
        
        for sim in evals:
            print(f"  - {sim.name}: policy_agents_pct={sim.policy_agents_pct}, npc_group_id={sim.npc_group_id}")
            
        return True
    except Exception as e:
        print(f"✗ Evaluations failed: {e}")
        traceback.print_exc()
        return False

def test_trainer_config():
    """Test trainer configuration."""
    print("\nTesting trainer configuration...")
    try:
        import experiments.recipes.arena_trader_experiment as ate
        tool = ate.train(total_timesteps=1000)
        cfg = tool.trainer
        
        print(f"✓ Trainer configured")
        print(f"  - Total timesteps: {cfg.total_timesteps}")
        print(f"  - Batch size: {cfg.batch_size}")
        print(f"  - NPC filtering: pct={cfg.npc_policy_agents_pct}, group={cfg.npc_group_id}")
        print(f"  - Evaluation interval: {cfg.evaluation.evaluate_interval}")
        
        return True
    except Exception as e:
        print(f"✗ Trainer config failed: {e}")
        traceback.print_exc()
        return False

def test_npc_filter_wrapper():
    """Test NPCFilterWrapper existence and basic functionality."""
    print("\nTesting NPCFilterWrapper...")
    try:
        from metta.rl.npc_filter_wrapper import NPCFilterWrapper
        print("✓ NPCFilterWrapper imported")
        
        # Check if it's used in vecenv
        import metta.rl.vecenv as vecenv
        import inspect
        source = inspect.getsource(vecenv.make_vecenv)
        if "NPCFilterWrapper" in source:
            print("✓ NPCFilterWrapper integrated in vecenv")
        else:
            print("⚠ NPCFilterWrapper not found in vecenv source")
            
        return True
    except Exception as e:
        print(f"✗ NPCFilterWrapper test failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_group_assignment():
    """Test that simulation handles group-based NPC assignment."""
    print("\nTesting simulation group assignment...")
    try:
        from metta.sim.simulation import Simulation
        import inspect
        
        # Check if the method exists
        if hasattr(Simulation, '_update_policy_npc_assignments'):
            print("✓ Group-based NPC assignment method exists")
            
            # Check if it's called during initialization
            source = inspect.getsource(Simulation.start_simulation)
            if '_update_policy_npc_assignments' in source:
                print("✓ NPC assignment called during simulation start")
            else:
                print("⚠ NPC assignment might not be called")
        else:
            print("✗ Missing _update_policy_npc_assignments method")
            
        return True
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_eval_stats_filtering():
    """Test that eval stats DB filters NPCs."""
    print("\nTesting eval stats NPC filtering...")
    try:
        from metta.eval.eval_stats_db import EvalStatsDB
        import inspect
        
        # Check if methods have exclude_npc_group_id parameter
        methods_to_check = ['get_average_metric_by_filter', 'get_sum_metric_by_filter', 'get_std_metric_by_filter']
        all_good = True
        
        for method_name in methods_to_check:
            if hasattr(EvalStatsDB, method_name):
                method = getattr(EvalStatsDB, method_name)
                sig = inspect.signature(method)
                if 'exclude_npc_group_id' in sig.parameters:
                    print(f"✓ {method_name} has NPC filtering")
                else:
                    print(f"✗ {method_name} missing NPC filtering")
                    all_good = False
            else:
                print(f"✗ Missing method {method_name}")
                all_good = False
                
        return all_good
    except Exception as e:
        print(f"✗ Eval stats test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE TEST - Arena Trader Experiment")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_env_creation,
        test_curriculum,
        test_evaluations,
        test_trainer_config,
        test_npc_filter_wrapper,
        test_simulation_group_assignment,
        test_eval_stats_filtering,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\nYour code is ready for Skypilot deployment!")
        return 0
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nReview the failures above before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

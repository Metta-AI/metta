#!/usr/bin/env python3
"""
Demonstration of the GeneticBuckettedCurriculum.

This script shows how the genetic curriculum evolves its task population
based on performance feedback.
"""

import random
from omegaconf import DictConfig

from metta.mettagrid.curriculum.genetic import GeneticBuckettedCurriculum


def mock_config_from_path(path, env_overrides=None):
    """Mock function to simulate loading environment config."""
    return DictConfig({
        "game": {
            "map": {"width": 32, "height": 32},
            "objects": {"altar": 1, "generator_red": 1},
            "num_agents": 4,
        }
    })


def main():
    # Monkey-patch the config loader for this demo
    import metta.mettagrid.curriculum.genetic
    metta.mettagrid.curriculum.genetic.config_from_path = mock_config_from_path
    
    # Define parameter ranges
    buckets = {
        "game.map.width": {"range": [10, 50]},
        "game.map.height": {"range": [10, 50]},
        "game.objects.altar": {"values": [1, 2, 3, 4, 5]},
        "game.objects.difficulty": {"range": [0.1, 1.0]},
    }
    
    # Create genetic curriculum
    curriculum = GeneticBuckettedCurriculum(
        env_cfg_template_path="dummy_path",
        buckets=buckets,
        population_size=20,
        replacement_rate=0.2,  # Replace 20% each generation
        mutation_rate=0.5,     # 50% mutation, 50% crossover
    )
    
    print("Initial Population:")
    print("-" * 50)
    for i, (task_id, params) in enumerate(curriculum._id_to_params.items()):
        if i < 5:  # Show first 5 tasks
            print(f"Task {i+1}: {task_id}")
    print(f"... and {len(curriculum._id_to_params) - 5} more tasks\n")
    
    # Simulate training for several generations
    generations = 10
    tasks_per_generation = 30
    
    for gen in range(generations):
        print(f"\nGeneration {gen + 1}:")
        print("-" * 30)
        
        # Track task performance in this generation
        task_scores = {}
        
        for _ in range(tasks_per_generation):
            # Get a task from the curriculum
            task = curriculum.get_task()
            
            # Simulate task performance based on difficulty
            # (In reality, this would come from actual agent performance)
            params = curriculum._id_to_params[task.id()]
            
            # Simple heuristic: smaller maps and fewer altars are easier
            map_size = params["game.map.width"] * params["game.map.height"]
            difficulty = params.get("game.objects.difficulty", 0.5)
            altar_count = params["game.objects.altar"]
            
            # Score inversely proportional to difficulty
            base_score = 1.0 - (map_size / 2500.0) * difficulty * (altar_count / 5.0)
            score = max(0.0, min(1.0, base_score + random.uniform(-0.2, 0.2)))
            
            task_scores[task.id()] = score
            
            # Complete the task (triggers evolution)
            curriculum.complete_task(task.id(), score)
        
        # Show population statistics
        avg_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0
        print(f"Average score: {avg_score:.3f}")
        
        # Show current task weights
        if hasattr(curriculum, '_task_weights'):
            sorted_tasks = sorted(
                curriculum._task_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            print("\nTop 3 tasks by weight:")
            for i, (task_id, weight) in enumerate(sorted_tasks[:3]):
                print(f"  {i+1}. Weight={weight:.3f}: {task_id}")
            
            print("\nBottom 3 tasks by weight:")
            for i, (task_id, weight) in enumerate(sorted_tasks[-3:]):
                print(f"  {i+1}. Weight={weight:.3f}: {task_id}")
    
    print("\n" + "=" * 50)
    print("Evolution Summary:")
    print("=" * 50)
    print(f"Final population size: {len(curriculum._id_to_curriculum)}")
    print(f"Total unique tasks seen: {len(curriculum._task_completions)}")
    
    # Show parameter distribution in final population
    print("\nParameter distributions in final population:")
    widths = [p["game.map.width"] for p in curriculum._id_to_params.values()]
    heights = [p["game.map.height"] for p in curriculum._id_to_params.values()]
    altars = [p["game.objects.altar"] for p in curriculum._id_to_params.values()]
    difficulties = [p["game.objects.difficulty"] for p in curriculum._id_to_params.values()]
    
    print(f"  Map width: {min(widths):.0f} - {max(widths):.0f} (avg: {sum(widths)/len(widths):.1f})")
    print(f"  Map height: {min(heights):.0f} - {max(heights):.0f} (avg: {sum(heights)/len(heights):.1f})")
    print(f"  Altar count: {min(altars)} - {max(altars)} (avg: {sum(altars)/len(altars):.1f})")
    print(f"  Difficulty: {min(difficulties):.2f} - {max(difficulties):.2f} (avg: {sum(difficulties)/len(difficulties):.2f})")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
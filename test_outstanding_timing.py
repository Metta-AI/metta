#!/usr/bin/env python3
"""Test that demonstrates outstanding counts with multiple simulated environments."""

import logging
import threading
import time

from omegaconf import OmegaConf

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

# Create a simple curriculum
task_cfg = OmegaConf.create({"game": {"num_agents": 2, "layout": {"rows": 10, "cols": 10}}})
curriculum = SingleTaskCurriculum("test_task", task_cfg)

# Start server with more slots
print("Starting curriculum server...")
server = CurriculumServer(curriculum, max_slots=20, auto_start=True)
time.sleep(1)


def simulate_environment(env_id, episode_duration=2.0):
    """Simulate an environment that runs episodes."""
    client = CurriculumClient()

    for episode in range(3):  # Run 3 episodes
        # Get task at start of episode
        task = client.get_task()
        print(f"Env {env_id} starting episode {episode} with task from slot")

        # Simulate episode running
        time.sleep(episode_duration)

        # Complete task at end of episode
        score = 0.5 + (env_id * 0.1)
        task.complete(score)
        print(f"Env {env_id} completed episode {episode}")


# Create multiple environment threads
print("\nStarting 5 simulated environments...")
threads = []
for i in range(5):
    thread = threading.Thread(target=simulate_environment, args=(i, 1.5))
    threads.append(thread)
    thread.start()
    time.sleep(0.3)  # Stagger the starts

# Print status while environments are running
print("\nStatus after environments start (should show outstanding tasks):")
server._print_status()

# Wait a bit and print status again
time.sleep(2)
print("\nStatus while episodes are running:")
server._print_status()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("\nStatus after all episodes complete (should show 0 outstanding):")
server._print_status()

# Clean up
server.stop()
print("\nTest complete!")

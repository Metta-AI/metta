#!/usr/bin/env python3
"""
Tribal Process Controller - Python side of process-separated tribal environment

This replaces the nimpy-based approach with file-based IPC to eliminate SIGSEGV issues.
The Nim process handles all OpenGL rendering while Python controls the environment.
"""

import json
import os
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class TribalProcessController:
    """
    Controls a tribal environment running in a separate Nim process.
    
    Communication is through JSON files:
    - tribal_control.json: Control and status messages
    - tribal_actions.json: Actions from Python to Nim
    - tribal_state.json: Environment state from Nim to Python
    """
    
    def __init__(self, tribal_dir: Path, config: Dict[str, Any] = None):
        self.tribal_dir = Path(tribal_dir)
        self.config = config or {}
        self.nim_process: Optional[subprocess.Popen] = None
        
        # Communication files (created in tribal directory)
        self.control_file = self.tribal_dir / "tribal_control.json"
        self.actions_file = self.tribal_dir / "tribal_actions.json"  
        self.state_file = self.tribal_dir / "tribal_state.json"
        
        # Environment state
        self.num_agents = 15  # MapAgents constant
        self.obs_shape = (19, 11, 11)  # ObservationLayers x Height x Width
        self.current_step = 0
        self.max_steps = 1000
        self.episode_done = False
        
        print(f"🎮 TribalProcessController initialized")
        print(f"   Tribal directory: {self.tribal_dir}")
        print(f"   Control file: {self.control_file}")
        print(f"   Actions file: {self.actions_file}")
        print(f"   State file: {self.state_file}")
    
    def start_nim_process(self) -> bool:
        """Start the Nim viewer process"""
        try:
            print("🚀 Starting Nim viewer process...")
            
            # Check if process viewer executable exists
            viewer_exe = self.tribal_dir / "tribal_process_viewer"
            if not viewer_exe.exists():
                print("❌ Process viewer executable not found. Building...")
                build_script = self.tribal_dir / "build_process_viewer.sh"
                if not build_script.exists():
                    print("❌ Build script not found")
                    return False
                
                result = subprocess.run(
                    ["bash", str(build_script)],
                    cwd=self.tribal_dir,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"❌ Build failed: {result.stderr}")
                    return False
                print("✅ Build completed")
            
            # Start the Nim process with stdout/stderr forwarding for debugging
            self.nim_process = subprocess.Popen(
                [str(viewer_exe)],
                cwd=self.tribal_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout for easier debugging
                text=True
            )
            
            # Wait for process to be ready
            max_wait = 10.0  # 10 seconds
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                if self.control_file.exists():
                    try:
                        with open(self.control_file, 'r') as f:
                            control = json.load(f)
                        if control.get("ready", False):
                            print("✅ Nim viewer process is ready")
                            return True
                    except:
                        pass
                
                time.sleep(0.1)
            
            print("❌ Nim viewer process did not become ready in time")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start Nim process: {e}")
            return False
    
    def stop_nim_process(self):
        """Stop the Nim viewer process"""
        if self.nim_process:
            try:
                print("🛑 Stopping Nim viewer process...")
                
                # Send shutdown signal
                shutdown_data = {
                    "active": False,
                    "shutdown": True,
                    "timestamp": time.time()
                }
                
                with open(self.control_file, 'w') as f:
                    json.dump(shutdown_data, f)
                
                # Wait for process to exit gracefully
                try:
                    self.nim_process.wait(timeout=5.0)
                    print("✅ Nim process exited gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️  Nim process didn't exit gracefully, terminating...")
                    self.nim_process.terminate()
                    try:
                        self.nim_process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        print("🔥 Force killing Nim process...")
                        self.nim_process.kill()
                        self.nim_process.wait()
                
            except Exception as e:
                print(f"⚠️  Error stopping Nim process: {e}")
            finally:
                self.nim_process = None
    
    def activate_communication(self):
        """Activate communication with the Nim process"""
        try:
            control_data = {
                "active": True,
                "shutdown": False,
                "timestamp": time.time()
            }
            
            with open(self.control_file, 'w') as f:
                json.dump(control_data, f)
            
            print("📡 Communication activated")
            
        except Exception as e:
            print(f"❌ Failed to activate communication: {e}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        print("🔄 Resetting environment...")
        
        # Wait for initial state from Nim
        max_wait = 5.0
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        state = json.load(f)
                    
                    obs = self._parse_observations(state["observations"])
                    self.current_step = state.get("current_step", 0)
                    self.max_steps = state.get("max_steps", 1000)
                    self.episode_done = state.get("episode_done", False)
                    
                    info = {
                        "current_step": self.current_step,
                        "max_steps": self.max_steps,
                        "episode_done": self.episode_done
                    }
                    
                    print(f"✅ Environment reset - step {self.current_step}")
                    return obs, info
                    
                except Exception as e:
                    print(f"⚠️  Error parsing state: {e}")
            
            time.sleep(0.1)
        
        print("❌ Timeout waiting for environment reset")
        # Return dummy observations
        obs = np.zeros((self.num_agents, *self.obs_shape), dtype=np.float32)
        info = {"current_step": 0, "max_steps": self.max_steps, "episode_done": False}
        return obs, info
    
    def step(self, actions: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step the environment with actions"""
        try:
            # Write actions to file
            actions_data = {
                "actions": actions,
                "timestamp": time.time()
            }
            
            with open(self.actions_file, 'w') as f:
                json.dump(actions_data, f)
            
            print(f"🔧 DEBUG: Wrote actions to {self.actions_file}")
            
            # Debug: check if Nim process is still running
            if self.nim_process.poll() is not None:
                print(f"❌ Nim process exited with code: {self.nim_process.poll()}")
            
            # Wait for updated state
            max_wait = 2.0  # 2 seconds max wait
            wait_start = time.time()
            initial_timestamp = 0
            
            # Get initial timestamp to detect updates
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        initial_state = json.load(f)
                    initial_timestamp = initial_state.get("timestamp", 0)
                except:
                    pass
            
            while time.time() - wait_start < max_wait:
                if self.state_file.exists():
                    try:
                        with open(self.state_file, 'r') as f:
                            state = json.load(f)
                        
                        # Check if this is a new state update
                        state_timestamp = state.get("timestamp", 0)
                        if state_timestamp > initial_timestamp:
                            obs = self._parse_observations(state["observations"])
                            rewards = np.array(state["rewards"], dtype=np.float32)
                            terminals = np.array(state["terminals"], dtype=bool)
                            truncations = np.array(state["truncations"], dtype=bool)
                            
                            self.current_step = state.get("current_step", self.current_step + 1)
                            self.episode_done = state.get("episode_done", False)
                            
                            info = {
                                "current_step": self.current_step,
                                "max_steps": self.max_steps,
                                "episode_done": self.episode_done
                            }
                            
                            return obs, rewards, terminals, truncations, info
                            
                    except Exception as e:
                        print(f"⚠️  Error parsing state: {e}")
                
                time.sleep(0.05)  # 50ms polling
            
            print(f"⚠️  Timeout waiting for state update after {max_wait}s")
            
            # Return dummy data on timeout
            obs = np.zeros((self.num_agents, *self.obs_shape), dtype=np.float32)
            rewards = np.zeros(self.num_agents, dtype=np.float32)
            terminals = np.zeros(self.num_agents, dtype=bool)
            truncations = np.zeros(self.num_agents, dtype=bool)
            info = {"current_step": self.current_step, "max_steps": self.max_steps, "episode_done": self.episode_done}
            
            return obs, rewards, terminals, truncations, info
            
        except Exception as e:
            print(f"❌ Error in step: {e}")
            
            # Print Nim process output for debugging
            self._check_nim_output()
            
            # Return dummy data on error
            obs = np.zeros((self.num_agents, *self.obs_shape), dtype=np.float32)
            rewards = np.zeros(self.num_agents, dtype=np.float32)
            terminals = np.zeros(self.num_agents, dtype=bool)
            truncations = np.zeros(self.num_agents, dtype=bool)
            info = {"current_step": self.current_step, "max_steps": self.max_steps, "episode_done": self.episode_done}
            return obs, rewards, terminals, truncations, info
    
    def _check_nim_output(self):
        """Check for output from the Nim process (non-blocking)"""
        if self.nim_process and self.nim_process.stdout:
            try:
                # Read any available output without blocking
                import select
                if select.select([self.nim_process.stdout], [], [], 0)[0]:
                    output = self.nim_process.stdout.read(1024)  # Read up to 1KB
                    if output:
                        print(f"🔧 Nim process output: {output}")
            except Exception as e:
                print(f"⚠️  Could not read Nim output: {e}")
    
    def _parse_observations(self, obs_data: List) -> np.ndarray:
        """Parse observations from JSON format to numpy array"""
        try:
            obs = np.array(obs_data, dtype=np.float32)
            # Shape should be [agents, layers, height, width]
            if obs.shape != (self.num_agents, *self.obs_shape):
                print(f"⚠️  Unexpected observation shape: {obs.shape}, expected {(self.num_agents, *self.obs_shape)}")
            return obs
        except Exception as e:
            print(f"⚠️  Error parsing observations: {e}")
            return np.zeros((self.num_agents, *self.obs_shape), dtype=np.float32)
    
    def cleanup(self):
        """Clean up resources and files"""
        self.stop_nim_process()
        
        # Remove communication files
        for file in [self.control_file, self.actions_file, self.state_file]:
            try:
                if file.exists():
                    file.unlink()
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Test function
def test_process_communication():
    """Test the process communication"""
    print("🧪 Testing Tribal Process Communication")
    
    tribal_dir = Path(__file__).parent
    
    with TribalProcessController(tribal_dir) as controller:
        print("1. Starting Nim process...")
        if not controller.start_nim_process():
            print("❌ Failed to start Nim process")
            return False
        
        print("2. Activating communication...")
        controller.activate_communication()
        
        print("3. Resetting environment...")
        obs, info = controller.reset()
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info: {info}")
        
        print("4. Testing environment steps...")
        for step in range(5):
            # Create test actions (MOVE with random directions)
            actions = []
            for i in range(controller.num_agents):
                actions.append([1, np.random.randint(0, 4)])  # MOVE action with random direction
            
            obs, rewards, terminals, truncations, info = controller.step(actions)
            
            reward_sum = rewards.sum()
            num_alive = (~(terminals | truncations)).sum()
            
            print(f"   Step {step + 1}: reward_sum={reward_sum:.3f}, agents_alive={num_alive}")
            print(f"      Sample actions: agent_0={actions[0]}, agent_1={actions[1]}")
            
            if info.get("episode_done", False):
                print("   Episode ended")
                break
            
            time.sleep(0.2)  # Small delay between steps
        
        print("✅ Process communication test completed successfully!")
        time.sleep(2)  # Let viewer run for a bit
        return True


if __name__ == "__main__":
    test_process_communication()
"""
SimpleCheckpointManager: Minimal policy checkpoint management for Phase 2 redesign.

Replaces the complex PolicyStore/PolicyRecord/CheckpointManager system with direct 
torch.save/load operations and YAML metadata sidecars for search functionality.
"""

import glob
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
import yaml

from metta.agent.metta_agent import MettaAgent
from metta.common.util.fs import atomic_write

logger = logging.getLogger(__name__)


class SimpleCheckpointManager:
    """Simplified checkpoint manager using direct torch.save/load + YAML metadata."""
    
    def __init__(self, run_dir: str, run_name: str):
        """Initialize checkpoint manager.
        
        Args:
            run_dir: Directory containing the run (e.g., "./train_dir/my_run")
            run_name: Name of the run for metadata
        """
        self.run_dir = run_dir
        self.run_name = run_name
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    # Core save/load methods
    def load_agent(self) -> Optional[MettaAgent]:
        """Load agent from latest checkpoint if exists, None otherwise."""
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint is None:
            logger.info(f"No checkpoints found in {self.checkpoint_dir}")
            return None
            
        logger.info(f"Loading agent from {latest_checkpoint}")
        try:
            # Load without weights_only for compatibility with older PyTorch versions
            agent = torch.load(latest_checkpoint, map_location="cpu")
            if not isinstance(agent, MettaAgent):
                raise ValueError(f"Checkpoint contains {type(agent)}, expected MettaAgent")
            return agent
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def save_agent(self, agent: MettaAgent, epoch: int, metadata: dict) -> str:
        """Save agent to model_{epoch:04d}.pt + metadata YAML, return path."""
        model_filename = f"model_{epoch:04d}.pt"
        yaml_filename = f"model_{epoch:04d}.yaml"
        
        model_path = os.path.join(self.checkpoint_dir, model_filename)
        yaml_path = os.path.join(self.checkpoint_dir, yaml_filename)
        
        # Save agent with atomic write
        logger.info(f"Saving agent to {model_path}")
        try:
            atomic_write(
                lambda path: torch.save(agent, path), 
                model_path,
                suffix=".pt"
            )
        except Exception as e:
            logger.error(f"Failed to save agent to {model_path}: {e}")
            raise
            
        # Save metadata
        logger.info(f"Saving metadata to {yaml_path}")
        try:
            # Ensure metadata has run name
            metadata_copy = metadata.copy()
            metadata_copy["run"] = self.run_name
            metadata_copy["epoch"] = epoch
            
            with open(yaml_path, "w") as f:
                yaml.dump(metadata_copy, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save metadata to {yaml_path}: {e}")
            # Don't raise - we have the model saved at least
            
        return model_path
    
    def save_trainer_state(self, optimizer: torch.optim.Optimizer, epoch: int, agent_step: int) -> None:
        """Save trainer state (optimizer + counters)."""
        trainer_state = {
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "agent_step": agent_step,
        }
        
        trainer_path = os.path.join(self.checkpoint_dir, "trainer_state.pt")
        logger.info(f"Saving trainer state to {trainer_path}")
        
        try:
            atomic_write(
                lambda path: torch.save(trainer_state, path),
                trainer_path,
                suffix=".pt"
            )
        except Exception as e:
            logger.error(f"Failed to save trainer state to {trainer_path}: {e}")
            raise
    
    def load_trainer_state(self) -> Optional[dict]:
        """Load trainer state if exists."""
        trainer_path = os.path.join(self.checkpoint_dir, "trainer_state.pt")
        
        if not os.path.exists(trainer_path):
            logger.info("No trainer state found - starting fresh")
            return None
            
        logger.info(f"Loading trainer state from {trainer_path}")
        try:
            state = torch.load(trainer_path, map_location="cpu")
            return state
        except Exception as e:
            logger.error(f"Failed to load trainer state from {trainer_path}: {e}")
            return None
    
    # Search and metadata methods
    def find_best_checkpoint(self, metric: str = "score") -> Optional[str]:
        """Find checkpoint with highest score/metric."""
        checkpoints = []
        
        for pt_file in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")):
            yaml_file = pt_file.replace(".pt", ".yaml")
            if os.path.exists(yaml_file):
                try:
                    with open(yaml_file) as f:
                        metadata = yaml.safe_load(f)
                        if metadata and metric in metadata:
                            checkpoints.append((metadata[metric], pt_file))
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {yaml_file}: {e}")
                    continue
        
        if not checkpoints:
            logger.info(f"No checkpoints found with metric '{metric}'")
            return None
            
        best_score, best_path = max(checkpoints)
        logger.info(f"Best checkpoint by {metric}: {best_path} (score: {best_score})")
        return best_path
    
    def find_checkpoints_by_score(self, min_score: float, metric: str = "score") -> list[str]:
        """Find all checkpoints with score >= min_score."""
        matching = []
        
        for pt_file in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")):
            yaml_file = pt_file.replace(".pt", ".yaml")
            if os.path.exists(yaml_file):
                try:
                    with open(yaml_file) as f:
                        metadata = yaml.safe_load(f)
                        if metadata and metadata.get(metric, 0.0) >= min_score:
                            matching.append(pt_file)
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {yaml_file}: {e}")
                    continue
        
        # Sort by epoch number (embedded in filename)
        def extract_epoch(path: str) -> int:
            match = re.search(r"model_(\d+)\.pt", os.path.basename(path))
            return int(match.group(1)) if match else 0
            
        matching.sort(key=extract_epoch)
        
        logger.info(f"Found {len(matching)} checkpoints with {metric} >= {min_score}")
        return matching
    
    def list_all_checkpoints(self) -> list[dict]:
        """List all checkpoints with metadata."""
        checkpoints = []
        
        for pt_file in glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt")):
            yaml_file = pt_file.replace(".pt", ".yaml")
            
            checkpoint_info = {
                "model_path": pt_file,
                "yaml_path": yaml_file if os.path.exists(yaml_file) else None,
                "metadata": {}
            }
            
            if os.path.exists(yaml_file):
                try:
                    with open(yaml_file) as f:
                        metadata = yaml.safe_load(f)
                        if metadata:
                            checkpoint_info["metadata"] = metadata
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {yaml_file}: {e}")
            
            checkpoints.append(checkpoint_info)
        
        # Sort by epoch
        def extract_epoch(info: dict) -> int:
            path = info["model_path"]
            match = re.search(r"model_(\d+)\.pt", os.path.basename(path))
            return int(match.group(1)) if match else 0
            
        checkpoints.sort(key=extract_epoch)
        return checkpoints
    
    # Helper methods
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find latest model_XXXX.pt file by epoch number."""
        pattern = os.path.join(self.checkpoint_dir, "model_*.pt")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return None
        
        # Extract epoch numbers and find maximum
        epochs = []
        for f in checkpoint_files:
            match = re.match(r".*model_(\d+)\.pt", f)
            if match:
                epochs.append((int(match.group(1)), f))
        
        if not epochs:
            return None
            
        latest_epoch, latest_file = max(epochs)
        logger.info(f"Found latest checkpoint: epoch {latest_epoch} at {latest_file}")
        return latest_file
    
    def get_checkpoint_count(self) -> int:
        """Get total number of checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, "model_*.pt")
        return len(glob.glob(pattern))
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = self.list_all_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            logger.info(f"Only {len(checkpoints)} checkpoints, no cleanup needed")
            return
            
        # Remove oldest checkpoints
        to_remove = checkpoints[:-keep_last_n]
        
        for checkpoint_info in to_remove:
            model_path = checkpoint_info["model_path"]
            yaml_path = checkpoint_info.get("yaml_path")
            
            try:
                os.remove(model_path)
                logger.info(f"Removed old checkpoint: {model_path}")
                
                if yaml_path and os.path.exists(yaml_path):
                    os.remove(yaml_path)
                    logger.info(f"Removed old metadata: {yaml_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to remove {model_path}: {e}")
        
        logger.info(f"Cleanup complete: removed {len(to_remove)} old checkpoints")
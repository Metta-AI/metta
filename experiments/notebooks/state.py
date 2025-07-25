"""State management utilities for tracking runs in notebooks."""

from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import subprocess
from experiments.types import TrainingJob


class RunState:
    """Manages state for tracking training runs in notebooks."""
    
    def __init__(self, 
                 wandb_run_names: Optional[List[str]] = None,
                 skypilot_job_ids: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize run state.
        
        Args:
            wandb_run_names: Pre-populated wandb run names
            skypilot_job_ids: Pre-populated sky job IDs
            metadata: Additional metadata from experiments
        """
        self.wandb_run_names = wandb_run_names or []
        self.skypilot_job_ids = skypilot_job_ids or []
        self.experiments = {}
        self.metadata = metadata or {}
        
        # Track all jobs launched in this session
        self.all_launched_jobs: List[Tuple[str, str]] = []
        self.training_jobs: List[TrainingJob] = []
        
        # Initialize with pre-populated jobs
        if self.wandb_run_names:
            for i, run_name in enumerate(self.wandb_run_names):
                job_id = self.skypilot_job_ids[i] if i < len(self.skypilot_job_ids) else None
                job = TrainingJob(
                    wandb_run_name=run_name,
                    skypilot_job_id=job_id,
                    notes='Pre-loaded from experiment',
                )
                self.training_jobs.append(job)
                
                if job_id:
                    self.all_launched_jobs.append((run_name, job_id))
                    
                self.experiments[run_name] = {
                    'job_id': job_id,
                    'config': {},
                    'notes': 'Pre-loaded from experiment',
                    'timestamp': self.metadata.get('created_at', 'Unknown')
                }
    
    def add_run(self, run_name: str, job_id: Optional[str] = None, 
                config: Optional[Dict] = None, notes: str = "") -> None:
        """Add a run to track in this session."""
        self.wandb_run_names.append(run_name)
        if job_id:
            self.skypilot_job_ids.append(job_id)
            self.all_launched_jobs.append((run_name, job_id))
        
        # Create TrainingJob
        job = TrainingJob(
            wandb_run_name=run_name,
            skypilot_job_id=job_id,
            notes=notes,
        )
        self.training_jobs.append(job)
        
        # Store experiment info
        self.experiments[run_name] = {
            'job_id': job_id,
            'config': config or {},
            'notes': notes,
            'timestamp': datetime.now()
        }
        print(f"Added run: {run_name}")
        if job_id:
            print(f"  Sky job: {job_id}")
    
    def list_runs(self) -> None:
        """List all runs in this session."""
        if not self.wandb_run_names:
            print("No runs tracked yet")
            return
        
        print(f"Tracking {len(self.wandb_run_names)} runs:")
        for i, run_id in enumerate(self.wandb_run_names):
            job_info = f" (job: {self.skypilot_job_ids[i]})" if i < len(self.skypilot_job_ids) else ""
            print(f"  {i+1}. {run_id}{job_info}")
        
        if self.all_launched_jobs:
            print(f"\nTotal jobs launched in this session: {len(self.all_launched_jobs)}")
    
    def kill_all_jobs(self) -> None:
        """Kill all Sky jobs launched in this session."""
        if not self.all_launched_jobs:
            print("No jobs to kill")
            return
        
        print(f"Killing {len(self.all_launched_jobs)} jobs...")
        killed = 0
        failed = 0
        
        for run_name, job_id in self.all_launched_jobs:
            try:
                result = subprocess.run(
                    ["sky", "cancel", "-y", job_id],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✓ Killed {job_id} ({run_name})")
                    killed += 1
                else:
                    print(f"✗ Failed to kill {job_id} ({run_name}): {result.stderr.strip()}")
                    failed += 1
            except Exception as e:
                print(f"✗ Error killing {job_id} ({run_name}): {e}")
                failed += 1
        
        print(f"\nSummary: {killed} killed, {failed} failed")


# Global instance that notebooks will use
_state: Optional[RunState] = None


def init_state(wandb_run_names: Optional[List[str]] = None,
               skypilot_job_ids: Optional[List[str]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> RunState:
    """Initialize the global run state.
    
    This is called automatically by generated notebooks.
    """
    global _state
    _state = RunState(wandb_run_names, skypilot_job_ids, metadata)
    return _state


def get_state() -> RunState:
    """Get the global run state instance."""
    global _state
    if _state is None:
        _state = RunState()
    return _state


# Convenience functions that operate on the global state
def add_run(run_name: str, job_id: Optional[str] = None, 
            config: Optional[Dict] = None, notes: str = "") -> None:
    """Add a run to track in this session."""
    get_state().add_run(run_name, job_id, config, notes)


def list_runs() -> None:
    """List all runs in this session."""
    get_state().list_runs()


def kill_all_jobs() -> None:
    """Kill all Sky jobs launched in this session."""
    get_state().kill_all_jobs()


# Direct access to state attributes
@property
def wandb_run_names() -> List[str]:
    """Get list of wandb run names."""
    return get_state().wandb_run_names


@property  
def skypilot_job_ids() -> List[str]:
    """Get list of sky job IDs."""
    return get_state().skypilot_job_ids


@property
def experiments() -> Dict[str, Any]:
    """Get experiments dictionary."""
    return get_state().experiments
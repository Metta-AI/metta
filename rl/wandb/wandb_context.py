import copy
import wandb
import os
import sys
from omegaconf import OmegaConf

class WandbContext:
    def __init__(self, cfg, resume=True, name=None, run_id=None, data_dir=None):
        self.cfg = cfg
        self.resume = resume
        self.name = name or cfg.wandb.name
        self.run_id = cfg.wandb.run_id or self.cfg.run or wandb.util.generate_id()
        self.run = None
        self.data_dir = data_dir or self.cfg.run_dir
        self.job_type = self._job_type()
    def __enter__(self):
        if not self.cfg.wandb.enabled:
            assert not self.cfg.wandb.track, "wandb.track won't work if wandb.enabled is False"
            return None

        cfg = copy.deepcopy(self.cfg)
        self.run = wandb.init(
            id=self.run_id,
            job_type=self.job_type,
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=False),
            group=self.cfg.wandb.group,
            allow_val_change=True,
            name=self.name,
            monitor_gym=True,
            save_code=True,
            resume=self.resume,
            tags=[
                "user:" + os.environ.get("METTA_USER", "unknown")
            ],
            settings=wandb.Settings(quiet=True)
        )

        OmegaConf.save(cfg, os.path.join(self.data_dir, "config.yaml"))
        wandb.save(os.path.join(self.data_dir, "*.log"), base_path=self.data_dir, policy="live")
        wandb.save(os.path.join(self.data_dir, "*.yaml"), base_path=self.data_dir, policy="live")

        return self.run


    def _job_type(self) -> str:
        """Detect job type based on which script is being run."""
        # Get the entry point script
        entry_point = sys.argv[0]
        
        # Check for module execution style (python -m tools.X)
        if entry_point.endswith('__main__.py') and len(sys.argv) > 1:
            module_path = sys.modules['__main__'].__package__
            if module_path and module_path.startswith('tools.'):
                tool_name = module_path.split('.')[-1]
                return tool_name
                
        # Check for direct script execution style (python tools/X.py)
        if 'tools/' in entry_point or 'tools\\' in entry_point:
            tool_name = os.path.basename(entry_point).replace('.py', '')
            return tool_name
            
        # Default job type if detection fails
        return "unknown"

    @staticmethod
    def make_run(cfg, resume=True, name=None):
        return WandbContext(cfg, resume, name).__enter__()

    @staticmethod
    def cleanup_run(run):
        if run:
            wandb.finish()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_run(self.run)

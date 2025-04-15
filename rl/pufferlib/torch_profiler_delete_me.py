import logging
import os
import time

import boto3
import torch.profiler

logger = logging.getLogger(__name__)

class TorchProfiler:
    def __init__(self, trainer_cfg):
        # self.cfg = cfg
        self.trainer_cfg = trainer_cfg
        self.profile_dir = cfg.profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)
        self.s3_client = boto3.client("s3")
        self.profiler = None
        self._wait_steps = 0
        self._warmup_steps = 0
        self.active = False
        self._step_count = 0
        self._min_steps = 20  # Minimum number of steps to collect
        self._start_epoch = None
        self._profile_filename = None
        self._total_steps = 0

    def should_profile(self, epoch):
        # If we're already profiling, continue until we have enough steps
        if self.active:
            return True
        # Otherwise, start profiling at the configured interval
        return (
            self.cfg.trainer.profiler_interval_epochs > 0 and 
            epoch % self.cfg.trainer.profiler_interval_epochs == 0
        )

    def start_profiling(self, epoch, wait_steps=0, warmup_steps=0):
        """Start a profiling session with optional wait and warmup steps.
        
        Args:
            epoch: Current epoch number
            wait_steps: Number of steps to wait before starting to record
            warmup_steps: Number of steps to warm up before starting to record
        """
        if not self.should_profile(epoch):
            return None

        # If we're already profiling, just return the existing profiler
        if self.active:
            return self.profiler

        self._wait_steps = wait_steps
        self._warmup_steps = warmup_steps
        self.active = True
        self._step_count = 0
        self._total_steps = 0
        self._start_epoch = epoch

        self._profile_filename = f"trace_{os.path.basename(self.cfg.run_dir)}_epoch_{epoch}.json"
        logger.info(f"Starting torch profiler for epoch {epoch}, saving to {self._profile_filename}")

        def save_profile(prof):
            try:
                output_path = os.path.join(self.profile_dir, self._profile_filename)
                logger.info(f"on_trace_ready: Exporting profile trace to {output_path}")
                prof.export_chrome_trace(output_path)
                logger.info(f"on_trace_ready: Successfully exported profile trace to {output_path}")
                s3_location = f"torch_traces/{os.path.basename(self.cfg.run_dir)}/{self._profile_filename}"
                # self.s3_client.upload_file(output_path, "softmax-public", s3_location) # pause uploading to s3 for now
            except Exception as e:
                logger.error(f"on_trace_ready: Error exporting profile trace: {e}", exc_info=True)

        # Create a schedule that includes wait, warmup, and active phases
        def schedule_fn(step):
            self._total_steps = step
            if step < wait_steps:
                logger.info(f"Profiler waiting (step {step}/{wait_steps})")
                return torch.profiler.ProfilerAction.NONE
            elif step < wait_steps + warmup_steps:
                logger.info(f"Profiler warming up (step {step}/{wait_steps + warmup_steps})")
                return torch.profiler.ProfilerAction.WARMUP
            else:
                logger.info(f"Profiler recording (step {step})")
                return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=save_profile,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=schedule_fn,
        )


        
        self.profiler.__enter__()
        return self.profiler

    def step(self):
        """Step the profiler if it's active."""
        if self.profiler and self.active:
            self.profiler.step()
            self._step_count += 1
            if self._total_steps >= self._wait_steps + self._warmup_steps:
                logger.info(f"Profiler active step {self._step_count} (total steps: {self._total_steps})")

    def stop_profiling(self):
        """Stop the current profiling session."""
        if self.profiler:
            try:
                # If we haven't collected enough steps, log a warning but don't stop profiling
                if self._step_count < self._min_steps:
                    logger.warning(
                        f"Profiler has only collected {self._step_count} steps (total steps: {self._total_steps}). "
                        f"Continuing profiling into next epoch to collect more data."
                    )
                    return
                
                logger.info(f"Profiler completed with {self._step_count} active steps from epoch {self._start_epoch}")
                self.profiler.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error exiting profiler: {e}")
            finally:
                logger.info("Torch profiler session completed")
                self.profiler = None
                self.active = False
                self._start_epoch = None
                self._profile_filename = None
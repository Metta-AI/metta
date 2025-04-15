import logging
import os
import time

import boto3
import torch.profiler

logger = logging.getLogger(__name__)

class TorchProfiler:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.profile_dir = os.path.join(self.run_dir, "torch_traces")
        os.makedirs(self.profile_dir, exist_ok=True)
        self.s3_client = boto3.client("s3")
        self.profiler = None
        self.active = False
        self._step_count = 0
        self._start_epoch = None
        self._wait_steps = 1 # we can add these four steps to the trainer config if we want more flexibility
        self._warmup_steps = 3
        self._active_steps = 1
        self._repeat_steps = 0
        self._all_steps_count = self._wait_steps + self._warmup_steps + self._active_steps + self._repeat_steps       
        self._profile_filename = None

    def setup_profiler(self, epoch):
        """Start a profiling session with optional wait and warmup steps.
        
        Args:
            epoch: Current epoch number
            wait_steps: Number of steps to wait before starting to record
            warmup_steps: Number of steps to warm up before starting to record
        """
        if self.active:
            if self.profiler is None:
                raise RuntimeError("Profiler attempted when it should be inactive.")
            return

        self.active = True
        self._step_count = 0
        self._start_epoch = epoch
        logger.info(f"Starting torch profiler for epoch {epoch}.")
        
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=self.save_profile,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(
                wait=self._wait_steps, 
                warmup=self._warmup_steps, 
                active=self._active_steps, 
                repeat=self._repeat_steps,
            ),
        )

    def step(self):
        """Step the profiler if it's active."""
        self._step_count += 1
        if self._step_count > self._all_steps_count:
            try:
                self.profiler.__exit__(None, None, None)
                logger.info("Profiler exited successfully.")
            except Exception as e:
                logger.error(f"Error exiting profiler: {e}")
            self.active = False
            self.profiler = None
            return
        self.active_step_count = self._step_count - self._wait_steps - self._warmup_steps
        if self.active_step_count >= 0:
            self._profile_filename = f"trace_{os.path.basename(self.run_dir)}_epoch_{self._start_epoch}_{self.active_step_count}.json"
        self.profiler.step()

    def save_profile(self, prof):
        try:
            output_path = os.path.join(self.profile_dir, self._profile_filename)
            logger.info(f"on_trace_ready: Exporting profile trace to {output_path}") # delete this
            prof.export_chrome_trace(output_path)
            logger.info(f"on_trace_ready: Successfully exported profile trace to {output_path} at epoch {self._start_epoch}.")
            s3_location = f"torch_traces/{os.path.basename(self.run_dir)}/{self._profile_filename}"
            # compress json file before uploading
            
            # self.s3_client.upload_file(output_path, "softmax-public", s3_location) # pause uploading to s3 while testing
            logger.info(f"on_trace_ready: Successfully uploaded profile trace to s3 at epoch {self._start_epoch}.")
        except Exception as e:
            logger.error(f"on_trace_ready: Error exporting profile trace: {e}", exc_info=True)
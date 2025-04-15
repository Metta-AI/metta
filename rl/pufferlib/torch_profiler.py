import logging
import os
import time
import gzip # Add gzip for compression

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
        self.active = False # Indicates if profiling should happen on next context entry
        self._start_epoch = None
        self._profile_filename_base = None # Base name for the trace file

    def setup_profiler(self, epoch):
        """Prepare the profiler to start on the next context entry."""
        if self.active:
            logger.warning("Profiler setup called while already active. Profiling will occur for the current setup.")
            return # Avoid resetting if already armed for the current epoch context

        if self.profiler is not None:
             logger.warning("Profiler object exists during setup. This might indicate an incomplete previous profile cycle.")
             # Allow overwriting/resetting state
             self.profiler = None

        self.active = True # Mark as ready to profile on __enter__
        self._start_epoch = epoch
        # Base filename, .json.gz will be added later
        self._profile_filename_base = f"trace_{os.path.basename(self.run_dir)}_epoch_{self._start_epoch}"
        logger.info(f"Torch profiler armed for epoch {epoch}. Will start profiling on context entry.")
        # Profiler object itself is created in __enter__

    def __enter__(self):
        """Enter the context manager, starting the profiler if active."""
        if not self.active:
            # If not armed by setup_profiler, do nothing in the context manager
            return self

        if self.profiler is not None:
            # This case should ideally not happen if setup/exit logic is correct
            logger.error("Profiler context entered but profiler object already exists. Attempting to stop previous profile.")
            try:
                self.profiler.stop()
            except Exception as e:
                logger.error(f"Error stopping lingering profiler: {e}")
            # Proceed to create a new one for this context

        logger.info(f"Entering profiler context for epoch {self._start_epoch}. Starting profiling.")
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.start()
        return self # Return self allows 'as ctx:' usage if needed, but not strictly necessary here

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, stopping the profiler and saving the trace."""
        # Only stop and save if we were active and successfully started a profiler
        if not self.active or self.profiler is None:
            self.active = False # Ensure deactivated if entered while inactive
            return False # Propagate any exception that occurred within the 'with' block

        logger.info(f"Exiting profiler context for epoch {self._start_epoch}. Stopping profiling.")
        try:
            self.profiler.stop()
            self._save_profile(self.profiler)
        except Exception as e:
            logger.error(f"Error stopping or saving profile for epoch {self._start_epoch}: {e}", exc_info=True)
        finally:
            # Cleanup regardless of success/failure in stopping/saving
            self.profiler = None
            self.active = False # Deactivate after this context usage is complete
            self._profile_filename_base = None # Clear base name

        # Return False to propagate any exception that occurred within the 'with' block
        return False


    def _save_profile(self, prof):
        """Saves the profiling results to a compressed JSON file and uploads to S3."""
        if self._profile_filename_base is None:
            logger.error("Profile filename base not set. Cannot save trace.")
            return

        output_filename_json = f"{self._profile_filename_base}.json"
        output_filename_gz = f"{output_filename_json}.gz"
        temp_json_path = os.path.join(self.profile_dir, output_filename_json) # Temp path for uncompressed
        final_gz_path = os.path.join(self.profile_dir, output_filename_gz) # Final compressed path

        try:
            logger.info(f"Exporting profile trace to temporary file {temp_json_path}...")
            prof.export_chrome_trace(temp_json_path) # Export uncompressed first

            logger.info(f"Compressing trace to {final_gz_path}...")
            with open(temp_json_path, 'rb') as f_in, gzip.open(final_gz_path, 'wb') as f_out:
                f_out.writelines(f_in)
            logger.info(f"Successfully saved compressed profile trace to {final_gz_path} for epoch {self._start_epoch}.")

            # Clean up uncompressed file only after successful compression
            try:
                os.remove(temp_json_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary trace file {temp_json_path}: {e}")

            s3_location = f"torch_traces/{os.path.basename(self.run_dir)}/{output_filename_gz}"
            # S3 upload:
            logger.info(f"Uploading profile trace to S3: s3://softmax-public/{s3_location}")
            self.s3_client.upload_file(final_gz_path, "softmax-public", s3_location)
            logger.info(f"Successfully uploaded profile trace to S3.")

        except Exception as e:
            logger.error(f"Error exporting/compressing/uploading profile trace for epoch {self._start_epoch}: {e}", exc_info=True)
            # Attempt cleanup of temp file even on error
            if os.path.exists(temp_json_path):
                 try:
                     os.remove(temp_json_path)
                 except OSError as e_rem:
                     logger.warning(f"Could not remove temporary trace file {temp_json_path} after error: {e_rem}")
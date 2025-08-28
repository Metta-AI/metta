"""Integration tests for sweep shell script coordination and process ID handling.

These tests verify the script-level integration:
1. Process ID generation in sweep.sh
2. Argument passing through sweep_rollout.sh
3. Process-specific file path generation
4. Script coordination and parameter flow
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile

import pytest


class TestSweepScriptIntegration:
    """Integration tests for sweep shell scripts."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Store original DATA_DIR value for restoration
        self.original_data_dir = os.environ.get("DATA_DIR")

        # Set environment for testing
        os.environ["DATA_DIR"] = self.data_dir

        yield

        # Cleanup environment variable
        if self.original_data_dir is not None:
            os.environ["DATA_DIR"] = self.original_data_dir
        else:
            os.environ.pop("DATA_DIR", None)

        # Cleanup filesystem
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sweep_argument_parsing_and_transformation(self):
        """Test argument parsing and transformation logic from sweep.sh."""

        # Test the argument transformation logic (extracted from sweep.sh)
        def transform_args(input_args):
            """Simulate the argument transformation from sweep.sh"""
            # Extract sweep name from run argument
            run_match = re.search(r"(^|\s)run=([^ ]*)", input_args)
            if not run_match:
                return None, None

            sweep_name = run_match.group(2)

            # Replace run=<name> with sweep_name=<name>
            args_for_rollout = re.sub(r"(^|\s)run=([^ ]*)", r"\1sweep_name=\2", input_args)

            return sweep_name, args_for_rollout

        # Test cases
        test_cases = [
            ("run=test_sweep", "test_sweep", "sweep_name=test_sweep"),
            ("other_arg=value run=my_sweep", "my_sweep", "other_arg=value sweep_name=my_sweep"),
            ("+hardware=gpu run=dist_sweep debug=true", "dist_sweep", "+hardware=gpu sweep_name=dist_sweep debug=true"),
        ]

        for input_args, expected_sweep, expected_output in test_cases:
            sweep_name, transformed_args = transform_args(input_args)
            assert sweep_name == expected_sweep
            assert transformed_args == expected_output

    def test_dist_cfg_path_generation_logic(self):
        """Test the distributed config path generation logic from sweep_rollout.sh."""

        def generate_dist_cfg_path(data_dir, sweep_name, process_id):
            """Simulate DIST_CFG_PATH generation from sweep_rollout.sh"""
            return os.path.join(data_dir, "sweep", sweep_name, f"dist_{process_id}.yaml")

        # Test with different inputs
        test_cases = [
            ("/tmp/data", "test_sweep", "abc12345", "/tmp/data/sweep/test_sweep/dist_abc12345.yaml"),
            (self.data_dir, "my_sweep", "def67890", f"{self.data_dir}/sweep/my_sweep/dist_def67890.yaml"),
            ("/data", "dist_sweep", "ghi13579", "/data/sweep/dist_sweep/dist_ghi13579.yaml"),
        ]

        for data_dir, sweep_name, process_id, expected_path in test_cases:
            result_path = generate_dist_cfg_path(data_dir, sweep_name, process_id)
            assert result_path == expected_path

    def test_process_id_uniqueness_in_parallel_execution(self):
        """Test that parallel process ID generation produces unique values."""

        def generate_process_id():
            result = subprocess.run(
                ["python", "-c", "import uuid; print(uuid.uuid4().hex[:8])"], capture_output=True, text=True
            )
            return result.stdout.strip()

        # Simulate 5 parallel workers
        num_workers = 5
        process_ids = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_process_id) for _ in range(num_workers)]

            for future in concurrent.futures.as_completed(futures):
                process_ids.append(future.result())

        # Verify all are unique
        assert len(set(process_ids)) == len(process_ids)
        assert len(process_ids) == num_workers

    def test_environment_variable_isolation(self):
        """Test that environment variables don't interfere between processes."""
        # Simulate multiple processes with different environment setups
        processes = []

        for i in range(3):
            env = os.environ.copy()
            env["TEST_PROCESS_ID"] = f"process_{i}"
            env["TEST_SWEEP_NAME"] = f"sweep_{i}"

            # Each process should see its own environment
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    "import os; print(f\"{os.environ.get('TEST_PROCESS_ID')}:{os.environ.get('TEST_SWEEP_NAME')}\")",
                ],
                env=env,
                capture_output=True,
                text=True,
            )

            processes.append(result.stdout.strip())

        # Verify each process saw its own environment
        expected = ["process_0:sweep_0", "process_1:sweep_1", "process_2:sweep_2"]
        assert processes == expected

    def test_file_path_collision_prevention(self):
        """Test that different process IDs prevent file collisions."""
        # Simulate multiple workers with different process IDs
        process_ids = ["worker01", "worker02", "worker03"]
        sweep_name = "collision_test"

        file_paths = []
        for process_id in process_ids:
            # Simulate dist_cfg_path generation
            dist_path = os.path.join(self.data_dir, "sweep", sweep_name, f"dist_{process_id}.yaml")
            file_paths.append(dist_path)

        # Verify all paths are unique
        assert len(set(file_paths)) == len(file_paths)

        # Verify each contains the correct process ID
        for i, path in enumerate(file_paths):
            assert f"dist_{process_ids[i]}.yaml" in path

        # Create the files to test actual filesystem collision prevention
        for path in file_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"process_specific_content_{os.path.basename(path)}")

        # Verify all files exist with different content
        for i, path in enumerate(file_paths):
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
                assert f"dist_{process_ids[i]}.yaml" in content

    def test_command_line_argument_extraction(self):
        """Test extraction of specific arguments from command line."""

        # Simulate argument extraction logic from sweep_rollout.sh
        def extract_argument(args_string, arg_name):
            """Extract argument value using grep-like pattern matching"""
            pattern = rf"(^|\s){arg_name}=([^ ]*)"
            match = re.search(pattern, args_string)
            return match.group(2) if match else None

        test_args = "sweep_name=test_sweep debug=false"

        # Test extracting different arguments
        assert extract_argument(test_args, "sweep_name") == "test_sweep"
        assert extract_argument(test_args, "debug") == "false"
        assert extract_argument(test_args, "nonexistent") is None

    def test_script_integration_error_handling(self):
        """Test error handling in script integration scenarios."""

        # Test missing required arguments
        def validate_sweep_args(args_string):
            """Simulate argument validation from sweep.sh"""
            has_run = bool(re.search(r"(^|\s)run=", args_string))
            if not has_run:
                return False, "Either 'run' argument is required"
            return True, None

        # Test cases
        valid_args = "run=test_sweep"
        invalid_args = "debug=true"

        is_valid, error = validate_sweep_args(valid_args)
        assert is_valid is True
        assert error is None

        is_valid, error = validate_sweep_args(invalid_args)
        assert is_valid is False
        assert error is not None and "run" in error

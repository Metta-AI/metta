"""Tests for job runner execution logic."""

import io

import pytest

from devops.job_runner import JobResult, LocalJob, RemoteJob


class FakePopen:
    """Minimal Popen fake that mimics stdout + poll/wait for LocalJob."""

    def __init__(self, cmd, cwd=None, stdout=None, stderr=None, env=None, preexec_fn=None, **kwargs):
        self.cmd = cmd
        self.pid = 12345
        self.returncode = None
        self._poll_count = 0
        self._max_polls_before_exit = 2
        self._exit_code = 0
        self._output_lines = []
        self._terminated = False

    def set_behavior(self, output_lines=None, exit_code=0, max_polls=2):
        """Configure the fake process behavior."""
        self._output_lines = output_lines or []
        self._exit_code = exit_code
        self._max_polls_before_exit = max_polls
        # Create BytesIO for stdout
        self.stdout = io.BytesIO(b"\n".join(line.encode() for line in self._output_lines))

    def poll(self):
        """Simulate polling - returns None while running, exit code when done."""
        if self._poll_count < self._max_polls_before_exit:
            self._poll_count += 1
            return None
        self.returncode = self._exit_code
        return self._exit_code

    def wait(self, timeout=None):
        """Wait for process to complete."""
        self.returncode = self._exit_code
        return self._exit_code

    def terminate(self):
        """Terminate the process."""
        self._terminated = True
        self.returncode = -15

    def kill(self):
        """Kill the process."""
        self._terminated = True
        self.returncode = -9


@pytest.fixture
def fake_popen():
    """Factory fixture for creating FakePopen instances."""
    instances = []

    def factory(output_lines=None, exit_code=0, max_polls=2):
        def popen_constructor(*args, **kwargs):
            proc = FakePopen(*args, **kwargs)
            proc.set_behavior(output_lines=output_lines, exit_code=exit_code, max_polls=max_polls)
            instances.append(proc)
            return proc

        return popen_constructor

    return factory, instances


def test_localjob_success(tmp_path, monkeypatch, fake_popen):
    """Test successful local job execution."""
    factory, instances = fake_popen

    output_lines = ["Starting job", "Processing...", "Complete", "Exit code: 0"]
    monkeypatch.setattr("subprocess.Popen", factory(output_lines=output_lines, exit_code=0))

    job = LocalJob(name="test_job", cmd=["echo", "hello"], timeout_s=10, log_dir=str(tmp_path), cwd=str(tmp_path))

    result = job.wait(stream_output=False)

    assert isinstance(result, JobResult)
    assert result.exit_code == 0
    assert result.success


def test_localjob_failure(tmp_path, monkeypatch, fake_popen):
    """Test local job that fails with non-zero exit code."""
    factory, instances = fake_popen

    output_lines = ["Error occurred", "Exit code: 1"]
    monkeypatch.setattr("subprocess.Popen", factory(output_lines=output_lines, exit_code=1))

    job = LocalJob(name="failing_job", cmd=["false"], timeout_s=10, log_dir=str(tmp_path), cwd=str(tmp_path))

    result = job.wait(stream_output=False)

    assert result.exit_code == 1
    assert not result.success


def test_localjob_timeout(tmp_path, monkeypatch, fake_popen):
    """Test that local job timeout is enforced."""
    factory, instances = fake_popen

    # Process that never completes (polls infinitely)
    monkeypatch.setattr("subprocess.Popen", factory(output_lines=[], exit_code=0, max_polls=10000))

    # Mock time to make timeout trigger quickly
    time_value = [0.0]

    def fake_time():
        time_value[0] += 1.0  # Advance 1 second per call
        return time_value[0]

    monkeypatch.setattr("time.time", fake_time)

    # Track if cancel was called
    cancel_called = []
    original_cancel = LocalJob.cancel

    def track_cancel(self):
        cancel_called.append(True)
        return original_cancel(self)

    monkeypatch.setattr(LocalJob, "cancel", track_cancel)

    job = LocalJob(name="timeout_job", cmd=["sleep", "1000"], timeout_s=2, log_dir=str(tmp_path), cwd=str(tmp_path))

    result = job.wait(stream_output=False)

    assert result.exit_code == 124  # Timeout exit code
    assert len(cancel_called) > 0  # Cancel was called


def test_localjob_writes_logs(tmp_path, monkeypatch, fake_popen):
    """Test that job output is written to log file."""
    factory, instances = fake_popen

    output_lines = ["Line 1", "Line 2", "Line 3"]
    monkeypatch.setattr("subprocess.Popen", factory(output_lines=output_lines, exit_code=0))

    job = LocalJob(name="log_test", cmd=["echo"], timeout_s=10, log_dir=str(tmp_path), cwd=str(tmp_path))

    job.wait(stream_output=False)

    # Verify log file exists and contains output
    log_path = tmp_path / "log_test.log"
    assert log_path.exists()

    log_content = log_path.read_text()
    for line in output_lines:
        assert line in log_content


def test_remotejob_completion_by_exit_marker(tmp_path, monkeypatch):
    """Test that remote job completion is detected via exit code marker."""
    logs = ["Job started", "Processing data", "Job finished", "Exit code: 0"]

    def fake_tail_job_log(job_id, lines=10000):
        return "\n".join(logs)

    # Mock tail_job_log where it's imported (in job_runner module)
    monkeypatch.setattr(
        "devops.job_runner.tail_job_log",
        fake_tail_job_log,
    )

    job = RemoteJob(
        name="remote_test",
        module="test.module",
        args=["arg=value"],
        log_dir=str(tmp_path),
        job_id="sky-job-123",
    )

    # Mark as submitted to bypass launch
    job._submitted = True

    assert job.is_complete() is True

    result = job.get_result()
    assert result is not None
    assert result.exit_code == 0
    assert result.success


def test_remotejob_not_complete_without_marker(tmp_path, monkeypatch):
    """Test that remote job without exit marker is not considered complete."""
    logs = ["Job started", "Still running..."]

    def fake_tail_job_log(job_id, lines=10000):
        return "\n".join(logs)

    monkeypatch.setattr(
        "devops.job_runner.tail_job_log",
        fake_tail_job_log,
    )

    job = RemoteJob(
        name="remote_running",
        module="test.module",
        args=[],
        log_dir=str(tmp_path),
        job_id="sky-job-456",
    )

    job._submitted = True

    assert job.is_complete() is False


def test_remotejob_get_logs_accumulates(tmp_path, monkeypatch):
    """Test that remote job logs accumulate correctly."""
    log_sequence = ["Line 1", "Line 1\nLine 2", "Line 1\nLine 2\nLine 3"]
    call_count = [0]

    def fake_tail_job_log(job_id, lines=10000):
        result = log_sequence[min(call_count[0], len(log_sequence) - 1)]
        call_count[0] += 1
        return result

    monkeypatch.setattr(
        "devops.job_runner.tail_job_log",
        fake_tail_job_log,
    )

    job = RemoteJob(
        name="log_accumulate",
        module="test.module",
        args=[],
        log_dir=str(tmp_path),
        job_id="sky-job-789",
    )

    job._submitted = True

    # Get logs multiple times - should accumulate
    logs1 = job.get_logs()
    assert "Line 1" in logs1

    logs2 = job.get_logs()
    assert "Line 1" in logs2
    assert "Line 2" in logs2

    logs3 = job.get_logs()
    assert "Line 1" in logs3
    assert "Line 2" in logs3
    assert "Line 3" in logs3


def test_jobresult_success_property():
    """Test JobResult.success property."""
    # Success case
    result_ok = JobResult(name="ok", exit_code=0, logs_path="/tmp/ok.log", duration_s=1.0)
    assert result_ok.success is True

    # Failure cases
    result_fail = JobResult(name="fail", exit_code=1, logs_path="/tmp/fail.log", duration_s=1.0)
    assert result_fail.success is False

    result_timeout = JobResult(name="timeout", exit_code=124, logs_path="/tmp/timeout.log", duration_s=1.0)
    assert result_timeout.success is False


def test_localjob_cancel_terminates_process(tmp_path, monkeypatch, fake_popen):
    """Test that cancel() properly terminates the process."""
    factory, instances = fake_popen

    monkeypatch.setattr("subprocess.Popen", factory(output_lines=[], exit_code=0, max_polls=1000))

    # Mock os.getpgid and os.killpg to avoid trying to kill real processes
    def fake_getpgid(pid):
        return pid  # Return same PID as process group ID

    def fake_killpg(pgid, sig):
        # Mark the fake process as terminated instead of actually killing
        for instance in instances:
            if hasattr(instance, "_terminated"):
                instance._terminated = True

    monkeypatch.setattr("os.getpgid", fake_getpgid)
    monkeypatch.setattr("os.killpg", fake_killpg)

    job = LocalJob(name="cancel_test", cmd=["sleep", "100"], timeout_s=10, log_dir=str(tmp_path), cwd=str(tmp_path))

    # Submit the job (in real code this happens in wait(), but we need the process)
    job.submit()

    # Cancel it
    job.cancel()

    # Verify the fake process was terminated
    assert len(instances) > 0
    assert instances[0]._terminated


def test_remotejob_handles_nonzero_exit(tmp_path, monkeypatch):
    """Test that remote job correctly reports non-zero exit codes."""
    logs = ["Job started", "Error occurred", "Exit code: 42"]

    def fake_tail_job_log(job_id, lines=10000):
        return "\n".join(logs)

    monkeypatch.setattr(
        "devops.job_runner.tail_job_log",
        fake_tail_job_log,
    )

    job = RemoteJob(
        name="remote_fail",
        module="test.module",
        args=[],
        log_dir=str(tmp_path),
        job_id="sky-job-fail",
    )

    job._submitted = True

    assert job.is_complete() is True

    result = job.get_result()
    assert result.exit_code == 42
    assert not result.success

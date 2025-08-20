# Asana Integration Action Test Utilities

This directory contains utilities for testing and verifying the behavior of the `pr_gh_to_asana.py` GitHub Action workflow using real workflow run data and VCR cassettes.

## Scripts

### 1. `download_action_logs.py`

Downloads logs and artifacts from GitHub Actions workflow runs for a given branch and workflow name.

**Usage:**

```sh
python download_action_logs.py <branch_name> [--workflow "Asana Integration"]
```

- Downloads logs and artifacts for all runs of the specified workflow on the given branch.
- Each run's data is saved in a directory named `run_<run_id>`.

### 2. `replay_and_test.py`

Replays a previously recorded workflow run using VCR in strict replay mode, verifying that `pr_gh_to_asana.py` produces the same REST activity as in the original run.

**Usage:**

```sh
python replay_and_test.py <run_directory>
```

- `<run_directory>` should be a directory produced by `download_action_logs.py` (e.g., `run_1234567890`).
- The script infers the run ID from the directory name, parses environment variables from the run log, sets up the VCR cassette, and runs `pr_gh_to_asana.py` in replay mode.
- If the script attempts any HTTP requests not present in the cassette, the test will fail.

## Workflow

1. **Download logs and artifacts:**
   - Run `download_action_logs.py` for your branch of interest.
   - Example:
     ```sh
     python download_action_logs.py hollander/asana-integration-z
     ```
   - This will create one or more `run_<run_id>` directories.

2. **Replay and verify:**
   - Run `replay_and_test.py` on a specific run directory.
   - Example:
     ```sh
     python replay_and_test.py run_1234567890
     ```
   - The script will report success if all HTTP interactions match the cassette, or failure if there are discrepancies.

## Requirements

- Python 3.11+
- The `gh` CLI must be installed and authenticated for `download_action_logs.py`.

## Setup

This directory contains a self-contained virtual environment to manage dependencies.

1. **Activate the virtual environment:**
   ```sh
   source .venv/bin/activate
   ```
2. **Install dependencies (if not already installed):**
   The required packages are listed in `requirements.txt`. If it's your first time or dependencies have changed, run:
   ```sh
   pip install -r requirements.txt
   ```
3. **Deactivate when done:**
    When you are finished, you can exit the virtual environment by running:
    ```sh
    deactivate
    ```

---

For questions or troubleshooting, check the output of each script for details on errors or missing dependencies.

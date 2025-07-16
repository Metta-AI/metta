import os
import re
import subprocess
import threading
from datetime import datetime

import ipywidgets as widgets
import wandb
import yaml
from IPython.display import IFrame, clear_output, display


def _load_available_environments():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "sim", "all.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    environments = []
    if "simulations" in config:
        for sim_config in config["simulations"].values():
            if "env" in sim_config:
                env_path = sim_config["env"]
                environments.append(env_path)
    return environments


class TrainingConfigurator:
    """Manages training configuration inputs."""

    def __init__(self):
        self.run_name_input = widgets.Text(
            value=f"{os.environ.get('USER', 'user')}.training.{datetime.now().strftime('%m-%d-%H%M')}",
            placeholder="Enter run name",
            description="Run Name:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

        self.num_gpus_input = widgets.IntSlider(
            value=1, min=1, max=8, step=1, description="Number of GPUs:", style={"description_width": "initial"}
        )

        self.num_cpus_input = widgets.IntSlider(
            value=1, min=1, max=8, step=1, description="Number of CPUs:", style={"description_width": "initial"}
        )

        self.use_spot_input = widgets.Checkbox(
            value=True, description="Use Spot Instance", disabled=False, indent=False
        )

        self.curriculum_input = widgets.Dropdown(
            options=_load_available_environments(),
            value=_load_available_environments()[0],
            description="Curriculum:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

        self.additional_args_input = widgets.Textarea(
            value="",
            placeholder="Additional training arguments (one per line)\ne.g., trainer.total_timesteps=1000000",
            description="Extra Args:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px", height="100px"),
        )

        self.cmd_preview_output = widgets.Output()
        self._setup_observers()
        self._update_command_preview()

    def _setup_observers(self):
        """Set up observers for input changes."""
        self.run_name_input.observe(self._update_command_preview, "value")
        self.num_gpus_input.observe(self._update_command_preview, "value")
        self.num_cpus_input.observe(self._update_command_preview, "value")
        self.use_spot_input.observe(self._update_command_preview, "value")
        self.curriculum_input.observe(self._update_command_preview, "value")
        self.additional_args_input.observe(self._update_command_preview, "value")

    def generate_launch_command(self):
        """Generate the launch command based on current inputs."""
        cmd = [
            "./devops/skypilot/launch.py",
            "train",
            f"run={self.run_name_input.value}",
            f"trainer.curriculum={self.curriculum_input.value}",
            f"--gpus={self.num_gpus_input.value}",
            f"--cpus={self.num_cpus_input.value}",
        ]

        if self.use_spot_input.value:
            cmd.append("--use-spot")

        if self.additional_args_input.value.strip():
            for line in self.additional_args_input.value.strip().split("\n"):
                if line.strip():
                    cmd.append(line.strip())

        return cmd

    def _update_command_preview(self, *args):
        """Update the command preview display."""
        with self.cmd_preview_output:
            clear_output(wait=True)
            cmd = self.generate_launch_command()
            print("Launch command:")
            print(" ".join(cmd))

    def display(self):
        return widgets.VBox(
            [
                widgets.HTML("<b>Training Configuration:</b>"),
                self.run_name_input,
                self.num_gpus_input,
                self.num_cpus_input,
                self.use_spot_input,
                self.curriculum_input,
                self.additional_args_input,
                self.cmd_preview_output,
            ]
        )


class JobLauncher:
    """Manages job launching."""

    def __init__(self, configurator):
        self.configurator = configurator
        self.launch_button = widgets.Button(
            description="Launch Training",
            disabled=False,
            button_style="primary",
            tooltip="Launch training job on SkyPilot",
            icon="rocket",
        )
        self.output = widgets.Output()
        self.job_id = None
        self.job_name = None
        self.is_launching = False

        self.launch_button.on_click(self._launch_training)

    def _launch_training(self, b):
        if self.is_launching:
            return

        self.is_launching = True
        self.launch_button.disabled = True  # Disable button during launch

        try:
            with self.output:
                clear_output(wait=True)  # Clear any pending output
                print("🚀 Launching training job...")

                cmd = self.configurator.generate_launch_command()
                self.job_name = self.configurator.run_name_input.value

                print(f"\nCommand: {' '.join(cmd)}")
                print("\n" + "=" * 50)

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath("__file__"))),
                    )

                    for line in process.stdout or []:
                        print(line, end="")
                        if "Job ID:" in line or "sky-" in line:
                            parts = line.split()
                            for part in parts:
                                if part.startswith("sky-") and "-" in part[4:]:
                                    self.job_id = part
                                    print(f"\nJob ID captured: {self.job_id}")

                    process.wait()

                    if process.returncode == 0:
                        print("\nJob launched successfully!")
                        if self.job_id:
                            print(f"Job ID: {self.job_id}")
                        print(f"Job Name: {self.job_name}")
                    else:
                        print(f"\nLaunch failed with return code: {process.returncode}")

                except Exception as e:
                    print(f"\nError launching job: {str(e)}")
        finally:
            self.is_launching = False
            self.launch_button.disabled = False

    def display(self):
        """Display the launcher widgets."""
        return widgets.VBox([self.launch_button, self.output])


class JobStatusMonitor:
    """Monitors SkyPilot job status."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.output = widgets.Output()
        self.refresh_button = widgets.Button(
            description="Refresh Status",
            disabled=False,
            button_style="info",
            tooltip="Refresh job status",
            icon="refresh",
        )
        self.auto_refresh_checkbox = widgets.Checkbox(value=True, description="Auto-refresh every 15s", disabled=False)

        self.refresh_timer = None
        self.is_checking = False
        self.refresh_button.on_click(self._on_refresh_clicked)
        self.auto_refresh_checkbox.observe(self._toggle_auto_refresh, names="value")

    def _on_refresh_clicked(self, b):
        """Handle refresh button click."""
        if not self.is_checking:
            if self.auto_refresh_checkbox.value:
                self._start_auto_refresh()
            else:
                self.check_job_status()

    def check_job_status(self):
        if self.is_checking:
            return

        self.is_checking = True

        try:
            with self.output:
                clear_output()
                print(f"Checking status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")

                try:
                    result = subprocess.run(["sky", "jobs", "queue", "--refresh"], capture_output=True, text=True)

                    if result.returncode == 0:
                        output_lines = result.stdout.strip().split("\n")

                        job_found = False
                        if self.launcher.job_name:
                            print(f"Looking for job: {self.launcher.job_name}\n")
                            print("All jobs:")
                            print(result.stdout)

                            for line in output_lines:
                                if self.launcher.job_name in line or (
                                    self.launcher.job_id and self.launcher.job_id in line
                                ):
                                    job_found = True
                                    print("\n✓ Found our job:")
                                    if output_lines and "ID" in output_lines[0]:
                                        print(output_lines[0])
                                    print(line)
                                    break

                            if not job_found:
                                print("\nJob not found in queue (may have completed)")
                        else:
                            print("No job launched yet")
                    else:
                        print(f"Error checking status: {result.stderr}")

                except Exception as e:
                    print(f"Error: {str(e)}")
        finally:
            self.is_checking = False

    def _toggle_auto_refresh(self, change):
        if change["new"]:
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _stop_auto_refresh(self):
        if self.refresh_timer:
            self.refresh_timer.cancel()
            self.refresh_timer = None

    def _start_auto_refresh(self):
        self._stop_auto_refresh()

        if self.auto_refresh_checkbox.value:
            self.check_job_status()
            self.refresh_timer = threading.Timer(15.0, self._start_auto_refresh)
            self.refresh_timer.start()

    def _auto_refresh(self):
        """Legacy method for compatibility."""
        self._start_auto_refresh()

    def display(self):
        """Display the status monitor widgets."""
        return widgets.VBox([widgets.HBox([self.refresh_button, self.auto_refresh_checkbox]), self.output])

    def start_monitoring(self):
        """Start monitoring if a job was launched."""
        if self.launcher.job_name:
            if self.auto_refresh_checkbox.value:
                self._start_auto_refresh()
            else:
                self.check_job_status()


class WandBConnector:
    def __init__(self):
        self.output = widgets.Output()
        self.api = None
        self.run = None

        self.specific_run_input = widgets.Text(
            value="",
            placeholder="Run ID",
            description="Run ID:",
            disabled=False,
            style={"description_width": "initial"},
        )

        self.connect_button = widgets.Button(
            description="Select W&B run",
            disabled=False,
            button_style="success",
            tooltip="Select run",
            icon="link",
        )

        self.connect_button.on_click(self._connect_to_wandb)

    def _connect_to_wandb(self, b):
        with self.output:
            clear_output()
            print("Connecting to W&B...")

            try:
                self.api = wandb.Api()

                run_path = f"metta-research/metta/{self.specific_run_input.value}"

                print(f"\nLooking for run: {run_path}")

                self.run = self.api.run(run_path)

                print(f"\nConnected to run: {self.run.name}")
                print(f"State: {self.run.state}")
                print(f"Start time: {self.run.created_at}")
                print(f"URL: {self.run.url}")

                if self.run.summary:
                    print("\nLatest metrics:")
                    for key, value in list(self.run.summary.items())[:10]:
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")

            except Exception as e:
                print(f"\nError connecting to W&B: {str(e)}")
                print("\nMake sure you are connected to W&B: `metta status`")

    def display(self):
        return widgets.VBox([widgets.HBox([self.specific_run_input]), self.connect_button, self.output])


class MetricsFetcher:
    def __init__(self, wandb_connector):
        self.wandb_connector = wandb_connector
        self.metrics_df = None

        self.fetch_last_n = widgets.IntText(
            value=1000,
            description="Last N points:",
            style={"description_width": "initial"},
            tooltip="Number of most recent data points to fetch",
        )

        self.fetch_sample_rate = widgets.IntSlider(
            value=1,
            min=1,
            max=100,
            description="Sample every:",
            style={"description_width": "initial"},
            tooltip="Sample every Nth data point (1 = all data)",
        )

        self.fetch_button = widgets.Button(
            description="Fetch Metrics", disabled=False, button_style="primary", icon="download"
        )

        self.output = widgets.Output()
        self.fetch_button.on_click(self._fetch_metrics)

    def _fetch_metrics(self, b):
        """Fetch metrics from W&B."""

        with self.output:
            clear_output()

            if not self.wandb_connector.run:
                print("Please connect to W&B first")
                return

            print(f"Fetching last {self.fetch_last_n.value} metrics from W&B...")
            print(f"Sampling every {self.fetch_sample_rate.value} data point(s)...")

            try:
                # Fetch all available metrics - no filtering
                self.metrics_df = self.wandb_connector.run.history(
                    samples=self.fetch_last_n.value
                    # No keys parameter = fetch everything
                )

                if self.fetch_sample_rate.value > 1:
                    self.metrics_df = self.metrics_df.iloc[:: self.fetch_sample_rate.value]

                print(f"\nFetched {len(self.metrics_df)} data points")

                if len(self.metrics_df) > 0:
                    print(f"\nColumns: {list(self.metrics_df.columns)}")
                    print(f"\nStep range: {self.metrics_df['_step'].min()} to {self.metrics_df['_step'].max()}")
                    print("\nSample data:")
                    display(self.metrics_df.head())

                    if "overview/reward" in self.metrics_df.columns:
                        print("\nReward statistics:")
                        print(f"  Mean: {self.metrics_df['overview/reward'].mean():.4f}")
                        print(f"  Std: {self.metrics_df['overview/reward'].std():.4f}")
                        print(f"  Min: {self.metrics_df['overview/reward'].min():.4f}")
                        print(f"  Max: {self.metrics_df['overview/reward'].max():.4f}")

            except Exception as e:
                print(f"\nError fetching metrics: {str(e)}")

    def display(self):
        return widgets.VBox(
            [
                widgets.HTML("<b>Fetch Options:</b>"),
                self.fetch_last_n,
                self.fetch_sample_rate,
                self.fetch_button,
                self.output,
            ]
        )

    def auto_fetch(self):
        if self.wandb_connector.run:
            self._fetch_metrics(None)


class ReplayViewer:
    """Manages replay viewing from W&B runs."""

    def __init__(self, wandb_connector):
        self.wandb_connector = wandb_connector
        self.replay_urls = []
        self.selected_replay_url = None

        self.output = widgets.Output()
        self.view_button = widgets.Button(
            description="Fetch Replays", disabled=False, button_style="primary", icon="video"
        )

        self.replay_selector = widgets.Dropdown(
            options=[],
            description="Select Replay:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="600px"),
            disabled=True,
        )

        self.view_button.on_click(self._fetch_replays)
        self.replay_selector.observe(self._on_replay_selected, names="value")

    def _fetch_replays(self, b):
        with self.output:
            clear_output()

            if not self.wandb_connector.run:
                print("Please connect to W&B first")
                return

            print("Looking for replay files in W&B run files...")

            try:
                # Get all files from the run
                files = self.wandb_connector.run.files()
                replay_files = []

                # Filter for replay HTML files
                for file in files:
                    if "media/html/replays/link_" in file.name and file.name.endswith(".html"):
                        replay_files.append(file)

                if replay_files:
                    print(f"\nFound {len(replay_files)} replay HTML files")

                    # Sort by step number (extracted from filename)
                    def get_step_from_filename(file):
                        match = re.search(r"link_(\d+)_", file.name)
                        return int(match.group(1)) if match else 0

                    replay_files.sort(key=get_step_from_filename)

                    # Clear previous replays
                    self.replay_urls = []

                    # Process files (limit to avoid too many downloads)
                    max_files = min(10, len(replay_files))
                    recent_files = replay_files[-max_files:]

                    print(f"Processing {len(recent_files)} most recent files...")

                    for file in recent_files:
                        try:
                            # Download and read the HTML file
                            with file.download(replace=True, root="/tmp") as f:
                                content = f.read()
                            match = re.search(r'<a[^>]+href="([^"]+)"', content)
                            if match:
                                href = match.group(1)
                                if href:
                                    step = get_step_from_filename(file)
                                    self.replay_urls.append(
                                        {"step": step, "url": href, "filename": file.name, "label": f"Step {step:,}"}
                                    )

                        except Exception as e:
                            print(f"Could not parse {file.name}: {e}")

                    if self.replay_urls:
                        print(f"\nSuccessfully extracted {len(self.replay_urls)} replay URLs")

                        # Update dropdown options
                        self.replay_selector.options = [
                            (replay["label"], idx) for idx, replay in enumerate(self.replay_urls)
                        ]
                        self.replay_selector.disabled = False

                        # Select the most recent replay
                        self.replay_selector.value = len(self.replay_urls) - 1

                        print(f"\nSelected most recent replay (step {self.replay_urls[-1]['step']:,})")
                    else:
                        print("\nCould not extract replay URLs from HTML files")
                else:
                    print("No replay HTML files found in run files")

            except Exception as e:
                print(f"Error accessing run files: {e}")

    def _on_replay_selected(self, change):
        if change["new"] is not None and 0 <= change["new"] < len(self.replay_urls):
            selected = self.replay_urls[change["new"]]
            self.selected_replay_url = selected["url"]

            with self.output:
                clear_output()
                print(f"Selected replay at step {selected['step']:,}")

    def display(self):
        return widgets.VBox(
            [widgets.HTML("<b>Replay Viewer:</b>"), self.view_button, self.replay_selector, self.output]
        )

    def display_iframe(self, width=1000, height=600):
        if self.selected_replay_url:
            print("Loading MettaScope viewer...")
            print(f"\nDirect link: {self.selected_replay_url}")
            display(IFrame(src=self.selected_replay_url, width=width, height=height))
        else:
            print("No replay selected. Please fetch replays first.")

    def auto_fetch(self):
        """Auto-fetch replays if connected."""
        if self.wandb_connector.run:
            self._fetch_replays(None)

import os
import re
import subprocess
from datetime import datetime

import ipywidgets as widgets
import wandb
import yaml
from IPython.display import IFrame, clear_output, display
from run_store import get_runstore


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

    @classmethod
    def create_widgets(cls):
        """Create configuration widgets."""
        widgets_dict = {
            "run_name_input": widgets.Textarea(
                value=f"{os.environ.get('USER', 'user')}.training.{datetime.now().strftime('%m-%d-%H%M')}",
                placeholder="Enter run name",
                description="Run Name:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="500px", height="30px", min_height="30px"),
                rows=1,
            ),
            "num_gpus_input": widgets.IntSlider(
                value=1, min=1, max=8, step=1, description="Number of GPUs:", style={"description_width": "initial"}
            ),
            "num_cpus_input": widgets.IntSlider(
                value=1, min=1, max=8, step=1, description="Number of CPUs:", style={"description_width": "initial"}
            ),
            "use_spot_input": widgets.Checkbox(
                value=True, description="Use Spot Instance", disabled=False, indent=False
            ),
            "curriculum_input": widgets.Dropdown(
                options=_load_available_environments(),
                value=_load_available_environments()[0],
                description="Curriculum:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="500px"),
            ),
            "additional_args_input": widgets.Textarea(
                value="",
                placeholder="Additional training arguments (one per line)\ne.g., trainer.total_timesteps=1000000",
                description="Extra Args:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="500px", height="100px"),
            ),
            "cmd_preview_output": widgets.Output(),
        }

        # Set up observers
        def update_preview(*args):
            cls._update_command_preview(widgets_dict)

        for key in [
            "run_name_input",
            "num_gpus_input",
            "num_cpus_input",
            "use_spot_input",
            "curriculum_input",
            "additional_args_input",
        ]:
            widgets_dict[key].observe(update_preview, "value")

        # Initial preview
        cls._update_command_preview(widgets_dict)

        return widgets_dict

    @classmethod
    def generate_launch_command(cls, widgets_dict):
        """Generate the launch command based on current inputs."""
        cmd = [
            "./devops/skypilot/launch.py",
            "train",
            f"run={widgets_dict['run_name_input'].value.strip()}",
            f"trainer.curriculum={widgets_dict['curriculum_input'].value}",
            f"--gpus={widgets_dict['num_gpus_input'].value}",
            f"--cpus={widgets_dict['num_cpus_input'].value}",
        ]

        if widgets_dict["use_spot_input"].value:
            cmd.append("--use-spot")

        if widgets_dict["additional_args_input"].value.strip():
            for line in widgets_dict["additional_args_input"].value.strip().split("\n"):
                if line.strip():
                    cmd.append(line.strip())

        return cmd

    @classmethod
    def _update_command_preview(cls, widgets_dict):
        """Update the command preview display."""
        with widgets_dict["cmd_preview_output"]:
            clear_output(wait=True)
            cmd = cls.generate_launch_command(widgets_dict)
            print("Launch command:")
            print(" ".join(cmd))

    @classmethod
    def display(cls, widgets_dict):
        return widgets.VBox(
            [
                widgets.HTML("<b>Training Configuration:</b>"),
                widgets_dict["run_name_input"],
                widgets_dict["num_gpus_input"],
                widgets_dict["num_cpus_input"],
                widgets_dict["use_spot_input"],
                widgets_dict["curriculum_input"],
                widgets_dict["additional_args_input"],
                widgets_dict["cmd_preview_output"],
            ]
        )


class JobLauncher:
    """Manages job launching."""

    @classmethod
    def create_widgets(cls, config_widgets):
        """Create launcher widgets."""
        widgets_dict = {
            "launch_button": widgets.Button(
                description="Launch Training",
                disabled=False,
                button_style="primary",
                tooltip="Launch training job on SkyPilot",
                icon="rocket",
            ),
            "output": widgets.Output(),
            "config_widgets": config_widgets,
            "state": {"is_launching": False, "job_id": None, "job_name": None},
        }

        # Set up click handler
        widgets_dict["launch_button"].on_click(lambda b: cls._launch_training(widgets_dict))

        return widgets_dict

    @classmethod
    def _launch_training(cls, widgets_dict):
        state = widgets_dict["state"]
        if state["is_launching"]:
            return

        state["is_launching"] = True
        widgets_dict["launch_button"].disabled = True

        try:
            with widgets_dict["output"]:
                clear_output(wait=True)
                print("🚀 Launching training job...")

                cmd = TrainingConfigurator.generate_launch_command(widgets_dict["config_widgets"])
                job_name = widgets_dict["config_widgets"]["run_name_input"].value.strip()
                state["job_name"] = job_name

                # Add run to RunStore
                run_store = get_runstore()
                command_args = " ".join(cmd[2:])  # Skip "./devops/skypilot/launch.py train"
                run_store.add_run(job_name, command_args=command_args)

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
                                    state["job_id"] = part
                                    print(f"\nJob ID captured: {state['job_id']}")
                                    # Update RunStore with sky job ID
                                    if job_name:
                                        run_store.refresh_run(job_name)

                    process.wait()

                    if process.returncode == 0:
                        print("\nJob launched successfully!")
                        if state["job_id"]:
                            print(f"Job ID: {state['job_id']}")
                        print(f"Job Name: {job_name}")
                    else:
                        print(f"\nLaunch failed with return code: {process.returncode}")

                except Exception as e:
                    print(f"\nError launching job: {str(e)}")
        finally:
            state["is_launching"] = False
            widgets_dict["launch_button"].disabled = False

    @classmethod
    def display(cls, widgets_dict):
        """Display the launcher widgets."""
        return widgets.VBox([widgets_dict["launch_button"], widgets_dict["output"]])


class WandBConnector:
    """Manages W&B connection and run selection."""

    @classmethod
    def create_widgets(cls):
        """Create W&B connector widgets."""
        widgets_dict = {
            "specific_run_input": widgets.Textarea(
                value="",
                placeholder="Run ID",
                description="Run ID:",
                disabled=False,
                style={"description_width": "initial"},
                layout=widgets.Layout(width="300px", height="30px", min_height="30px"),
                rows=1,
            ),
            "connect_button": widgets.Button(
                description="Select W&B run",
                disabled=False,
                button_style="success",
                tooltip="Select run",
                icon="link",
            ),
            "output": widgets.Output(),
            "state": {"api": None, "run": None},
        }

        # Set up click handler
        widgets_dict["connect_button"].on_click(lambda b: cls._connect_to_wandb(widgets_dict))

        return widgets_dict

    @classmethod
    def _connect_to_wandb(cls, widgets_dict):
        """Connect to W&B and select a run."""
        with widgets_dict["output"]:
            clear_output()
            print("Connecting to W&B...")

            try:
                widgets_dict["state"]["api"] = wandb.Api()
            except Exception as e:
                print(f"\nError connecting to W&B: {str(e)}")
                print("\nMake sure you are connected to W&B: `metta status`")
                return

            run_path = f"metta-research/metta/{widgets_dict['specific_run_input'].value.strip()}"
            print(f"\nLooking for run: {run_path}")

            try:
                widgets_dict["state"]["run"] = widgets_dict["state"]["api"].run(run_path)
                run = widgets_dict["state"]["run"]

                print(f"\nConnected to run: {run.name}")
                print(f"State: {run.state}")
                print(f"Start time: {run.created_at}")
                print(f"URL: {run.url}")

                # Add run to RunStore
                run_store = get_runstore()
                run_store.add_run(run.name or run.id)
                # Force update to get latest W&B data
                run_store.refresh_run(run.name or run.id)

                if run.summary:
                    print("\nLatest metrics:")
                    for key, value in list(run.summary.items())[:10]:
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
            except Exception as e:
                print(f"\nError connecting to run: {str(e)}")

    @classmethod
    def display(cls, widgets_dict):
        """Display the W&B connector widgets."""
        return widgets.VBox(
            [widgets.HBox([widgets_dict["specific_run_input"]]), widgets_dict["connect_button"], widgets_dict["output"]]
        )


class MetricsFetcher:
    """Fetches metrics from W&B runs."""

    @classmethod
    def create_widgets(cls):
        """Create metrics fetcher widgets."""
        widgets_dict = {
            "run_names_input": widgets.Textarea(
                value="",
                placeholder="Enter run names (one per line)",
                description="Run names:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="400px", height="100px"),
            ),
            "fetch_last_n": widgets.Textarea(
                value="1000",
                description="Last N points:",
                style={"description_width": "initial"},
                tooltip="Number of most recent data points to fetch",
                layout=widgets.Layout(width="200px", height="30px", min_height="30px"),
                rows=1,
            ),
            "fetch_sample_rate": widgets.IntSlider(
                value=1,
                min=1,
                max=100,
                description="Sample every:",
                style={"description_width": "initial"},
                tooltip="Sample every Nth data point (1 = all data)",
            ),
            "fetch_button": widgets.Button(
                description="Fetch Metrics", disabled=False, button_style="primary", icon="download"
            ),
            "output": widgets.Output(),
            "state": {"metrics_dfs": {}, "replay_urls": {}, "api": None},
        }

        # Set up click handler
        widgets_dict["fetch_button"].on_click(lambda b: cls._fetch_metrics(widgets_dict))

        return widgets_dict

    @classmethod
    def _fetch_metrics(cls, widgets_dict):
        """Fetch metrics from W&B for multiple runs."""
        with widgets_dict["output"]:
            clear_output()

            run_names = [
                name.strip() for name in widgets_dict["run_names_input"].value.strip().split("\n") if name.strip()
            ]
            if not run_names:
                print("Please enter at least one run name")
                return

            # Initialize W&B API if needed
            if not widgets_dict["state"]["api"]:
                try:
                    widgets_dict["state"]["api"] = wandb.Api()
                except Exception as e:
                    print(f"Error connecting to W&B: {str(e)}")
                    print("Make sure you are connected to W&B: `metta status`")
                    return

            print(f"Fetching metrics for {len(run_names)} runs...")
            print(f"Last {int(widgets_dict['fetch_last_n'].value.strip())} points per run")
            print(f"Sampling every {widgets_dict['fetch_sample_rate'].value} data point(s)\n")

            # Clear previous data
            widgets_dict["state"]["metrics_dfs"] = {}
            widgets_dict["state"]["replay_urls"] = {}

            for run_name in run_names:
                print(f"\nFetching: {run_name}")
                try:
                    run_path = f"metta-research/metta/{run_name}"
                    run = widgets_dict["state"]["api"].run(run_path)

                    # Fetch metrics
                    metrics_df = run.history(samples=int(widgets_dict["fetch_last_n"].value.strip()))

                    if widgets_dict["fetch_sample_rate"].value > 1:
                        metrics_df = metrics_df.iloc[:: widgets_dict["fetch_sample_rate"].value]

                    widgets_dict["state"]["metrics_dfs"][run_name] = metrics_df
                    print(f"  ✓ Fetched {len(metrics_df)} data points")

                    if len(metrics_df) > 0 and "overview/reward" in metrics_df.columns:
                        print(
                            f"  Reward: mean={metrics_df['overview/reward'].mean():.4f}, "
                            f"max={metrics_df['overview/reward'].max():.4f}"
                        )

                    # Also fetch replay URLs
                    print("  Fetching replay URLs...")
                    replay_urls = cls._fetch_replay_urls_for_run(run)
                    if replay_urls:
                        widgets_dict["state"]["replay_urls"][run_name] = replay_urls
                        print(f"  ✓ Found {len(replay_urls)} replay files")
                    else:
                        print("  ⚠ No replay files found")

                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")

            print(f"\nSuccessfully fetched data for {len(widgets_dict['state']['metrics_dfs'])} runs")

    @classmethod
    def _fetch_replay_urls_for_run(cls, run):
        """Fetch replay URLs for a single run."""
        try:
            files = run.files()
            replay_urls = []

            # Filter for replay HTML files
            replay_files = [f for f in files if "media/html/replays/link_" in f.name and f.name.endswith(".html")]

            if not replay_files:
                return []

            # Sort by step number
            def get_step_from_filename(file):
                match = re.search(r"link_(\d+)_", file.name)
                return int(match.group(1)) if match else 0

            replay_files.sort(key=get_step_from_filename)

            # Process files (limit to avoid too many)
            max_files = min(20, len(replay_files))
            recent_files = replay_files[-max_files:]

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
                            replay_urls.append(
                                {"step": step, "url": href, "filename": file.name, "label": f"Step {step:,}"}
                            )
                except Exception:
                    pass

            return replay_urls
        except Exception:
            return []

    @classmethod
    def display(cls, widgets_dict):
        """Display the metrics fetcher widgets."""
        return widgets.VBox(
            [
                widgets.HTML("<b>Fetch Metrics from W&B:</b>"),
                widgets_dict["run_names_input"],
                widgets_dict["fetch_last_n"],
                widgets_dict["fetch_sample_rate"],
                widgets_dict["fetch_button"],
                widgets_dict["output"],
            ]
        )

    @classmethod
    def auto_fetch(cls, widgets_dict, run_names=None):
        """Auto-fetch metrics for specified runs."""
        if run_names:
            widgets_dict["run_names_input"].value = "\n".join(run_names)
            cls._fetch_metrics(widgets_dict)


class ReplayViewer:
    """Manages replay viewing from W&B runs."""

    @classmethod
    def create_widgets(cls, fetcher_widgets=None):
        """Create replay viewer widgets."""
        widgets_dict = {
            "run_selector": widgets.Dropdown(
                options=[],
                description="Select Run:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="400px"),
                disabled=True,
            ),
            "replay_selector": widgets.Dropdown(
                options=[],
                description="Select Replay:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="600px"),
                disabled=True,
            ),
            "output": widgets.Output(),
            "fetcher_widgets": fetcher_widgets,
            "state": {"selected_run": None, "selected_replay_url": None},
        }

        # Set up event handlers
        widgets_dict["run_selector"].observe(lambda change: cls._on_run_selected(widgets_dict, change), names="value")
        widgets_dict["replay_selector"].observe(
            lambda change: cls._on_replay_selected(widgets_dict, change), names="value"
        )

        return widgets_dict

    @classmethod
    def _on_run_selected(cls, widgets_dict, change):
        """Handle run selection."""
        if change["new"] is None:
            return

        with widgets_dict["output"]:
            clear_output()

            run_name = change["new"]
            widgets_dict["state"]["selected_run"] = run_name

            # Get replay URLs from fetcher widgets
            if widgets_dict["fetcher_widgets"] and "replay_urls" in widgets_dict["fetcher_widgets"]["state"]:
                replay_urls = widgets_dict["fetcher_widgets"]["state"]["replay_urls"].get(run_name, [])

                if replay_urls:
                    print(f"Found {len(replay_urls)} replays for {run_name}")

                    # Update replay selector
                    widgets_dict["replay_selector"].options = [
                        (replay["label"], idx) for idx, replay in enumerate(replay_urls)
                    ]
                    widgets_dict["replay_selector"].disabled = False

                    # Select the most recent replay
                    widgets_dict["replay_selector"].value = len(replay_urls) - 1
                else:
                    print(f"No replays found for {run_name}")
                    widgets_dict["replay_selector"].options = []
                    widgets_dict["replay_selector"].disabled = True
            else:
                print("Please fetch metrics first to load replay URLs")
                widgets_dict["replay_selector"].options = []
                widgets_dict["replay_selector"].disabled = True

    @classmethod
    def _on_replay_selected(cls, widgets_dict, change):
        """Handle replay selection."""
        if change["new"] is None:
            return

        run_name = widgets_dict["state"]["selected_run"]
        if not run_name:
            return

        # Get replay URLs from fetcher widgets
        if widgets_dict["fetcher_widgets"] and "replay_urls" in widgets_dict["fetcher_widgets"]["state"]:
            replay_urls = widgets_dict["fetcher_widgets"]["state"]["replay_urls"].get(run_name, [])

            if 0 <= change["new"] < len(replay_urls):
                selected = replay_urls[change["new"]]
                widgets_dict["state"]["selected_replay_url"] = selected["url"]

                with widgets_dict["output"]:
                    clear_output()
                    print(f"Selected replay at step {selected['step']:,}")

    @classmethod
    def display(cls, widgets_dict):
        """Display the replay viewer widgets."""
        # Update run selector with available runs from fetcher
        if widgets_dict["fetcher_widgets"] and "replay_urls" in widgets_dict["fetcher_widgets"]["state"]:
            runs_with_replays = list(widgets_dict["fetcher_widgets"]["state"]["replay_urls"].keys())
            if runs_with_replays:
                widgets_dict["run_selector"].options = runs_with_replays
                widgets_dict["run_selector"].disabled = False
                # Auto-select first run
                if not widgets_dict["state"]["selected_run"] and runs_with_replays:
                    widgets_dict["run_selector"].value = runs_with_replays[0]

        return widgets.VBox(
            [
                widgets.HTML("<b>Replay Viewer:</b>"),
                widgets_dict["run_selector"],
                widgets_dict["replay_selector"],
                widgets_dict["output"],
            ]
        )

    @classmethod
    def display_iframe(cls, widgets_dict, width=1000, height=600):
        """Display the selected replay in an iframe."""
        if widgets_dict["state"]["selected_replay_url"]:
            print("Loading MettaScope viewer...")
            print(f"\nDirect link: {widgets_dict['state']['selected_replay_url']}")
            display(IFrame(src=widgets_dict["state"]["selected_replay_url"], width=width, height=height))
        else:
            print("No replay selected. Please fetch replays first.")

    @classmethod
    def update_from_fetcher(cls, widgets_dict):
        """Update the replay viewer with data from the fetcher."""
        # This is now handled automatically in display()
        cls.display(widgets_dict)

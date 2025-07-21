import json
import threading
from datetime import datetime
from functools import cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from notebooks.clients.sky_client import SkyPilotClient, SkyPilotJobData, SkyStatus
from notebooks.clients.wandb_client import RunConfig, WandBClient, WandBRunData, WandBStatus


class Run(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    sky: SkyPilotJobData | None = None
    wandb: WandBRunData | None = None
    config: RunConfig = Field(default_factory=RunConfig)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("run_id cannot be empty")
        return v.strip()

    @property
    def is_active(self) -> bool:
        return bool(
            (self.sky and self.sky.status == SkyStatus.RUNNING)
            or (self.wandb and self.wandb.status == WandBStatus.RUNNING)
        )


class RunStoreData(BaseModel):
    runs: dict[str, Run] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)


@cache
def get_runstore() -> "RunStore":
    """Get the singleton RunStore instance."""
    return RunStore()


class RunStore:
    """Stateless RunStore that persists all data to disk."""

    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".metta" / "run_store.json"

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._sky_client = SkyPilotClient()
        self._wandb_client = WandBClient()

    def _load(self) -> RunStoreData:
        """Load data from disk. Returns fresh data each time."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    return RunStoreData.model_validate(data)
            except Exception as e:
                print(f"Warning: Could not load run store: {e}")
        return RunStoreData()

    def _save(self, store_data: RunStoreData) -> None:
        """Save data to disk."""
        with self._lock:
            if self.storage_path.exists():
                backup_path = self.storage_path.with_suffix(".json.bak")
                self.storage_path.rename(backup_path)

            store_data.updated_at = datetime.now()
            data = store_data.model_dump(mode="json")

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def add_run(self, run_id: str, command_args: str | None = None) -> Run:
        run_id = run_id.strip()
        print(f"[RunStore] Adding run: {run_id}")

        # Load current data
        store_data = self._load()

        if run_id in store_data.runs:
            print(f"[RunStore] Run '{run_id}' already exists, updating...")
            run = store_data.runs[run_id]
            if command_args:
                run.config.command_args = command_args
                run.updated_at = datetime.now()
        else:
            print(f"[RunStore] Creating new run: {run_id}")
            config = RunConfig()
            if command_args:
                config.command_args = command_args

            run = Run(run_id=run_id, config=config)
            store_data.runs[run_id] = run

        self._save(store_data)
        print(f"[RunStore] Saved to {self.storage_path}")

        # Verify it was saved
        print(f"[RunStore] Total runs in store: {len(store_data.runs)}")

        return store_data.runs[run_id]

    def get_all(self) -> list[Run]:
        store_data = self._load()
        runs = list(store_data.runs.values())
        runs.sort(key=lambda r: r.updated_at, reverse=True)
        return runs

    def get_run(self, run_id: str) -> Run | None:
        store_data = self._load()
        return store_data.runs.get(run_id)

    def remove_run(self, run_id: str) -> bool:
        store_data = self._load()
        if run_id in store_data.runs:
            del store_data.runs[run_id]
            self._save(store_data)
            return True
        return False

    def load_in_all(self) -> None:
        store_data = self._load()
        sky_runs = self._sky_client.get_all()
        wandb_runs = {run_id: self._wandb_client.get_run(run_id) for run_id in self._wandb_client.list_all()}

        run_ids = set(sky_runs.keys()) | set(wandb_runs.keys())

        for run_id in run_ids:
            if run_id not in store_data.runs:
                store_data.runs[run_id] = Run(run_id=run_id)
            if run_id in sky_runs:
                store_data.runs[run_id].sky = sky_runs[run_id]
            if run_id in wandb_runs:
                store_data.runs[run_id].wandb = wandb_runs[run_id]
        self._save(store_data)

    def refresh_all(self) -> list[Run]:
        # Load current data
        store_data = self._load()

        # Get all run IDs
        run_ids = list(store_data.runs.keys())

        # Batch fetch from both sources
        sky_jobs = self._sky_client.get_jobs_by_names(run_ids)
        wandb_runs = self._wandb_client.get_runs_by_ids(run_ids)

        updated_runs = []

        # Update each run with fresh data
        for run_id in run_ids:
            run = store_data.runs[run_id]
            updated = False

            if run_id in sky_jobs:
                run.sky = sky_jobs[run_id]
                updated = True

            if run_id in wandb_runs:
                run.wandb = wandb_runs[run_id]
                updated = True

            if updated:
                run.updated_at = datetime.now()
                updated_runs.append(run)

        if updated_runs:
            self._save(store_data)

        return updated_runs

    def refresh_run(self, run_id: str) -> bool:
        print(f"[RunStore] Refreshing run: {run_id}")

        # Load current data
        store_data = self._load()

        if run_id not in store_data.runs:
            print(f"[RunStore] Run {run_id} not found in store")
            return False

        run = store_data.runs[run_id]
        updated = False

        print(f"[RunStore] Querying SkyPilot for: {run_id}")
        if sky_data := self._sky_client.get_job_by_name(run_id):
            print(f"[RunStore] Found SkyPilot data: {sky_data.status}")
            run.sky = sky_data
            updated = True
        else:
            print(f"[RunStore] No SkyPilot data found for: {run_id}")

        print(f"[RunStore] Querying W&B for: {run_id}")
        if wandb_data := self._wandb_client.get_run(run_id):
            print(f"[RunStore] Found W&B data: {wandb_data.status}")
            run.wandb = wandb_data
            updated = True
        else:
            print(f"[RunStore] No W&B data found for: {run_id}")

        if updated:
            run.updated_at = datetime.now()
            self._save(store_data)
            print(f"[RunStore] Updated run {run_id} and saved")
        else:
            print(f"[RunStore] No updates found for run {run_id}")

        return updated

    def to_widget(self, runs: list[Run] | None = None, max_rows: int = 10):
        """Create an interactive widget table for displaying runs."""
        import ipywidgets as widgets

        if runs is None:
            runs = self.get_all()

        # State variables
        self._filtered_runs = runs
        self._current_page = 0
        self._sort_column = "created"
        self._sort_ascending = False

        # Create UI elements
        # Search and filters row
        search_label = widgets.Label("Search:")
        search_input = widgets.Textarea(
            value="",
            placeholder="Search run ID...",
            layout=widgets.Layout(width="250px", height="30px"),
            style={"description_width": "initial"},
            continuous_update=True,  # Update as user types
            rows=1,
        )

        sky_filter = widgets.Dropdown(
            options=["All Sky Status", "RUNNING", "SUCCEEDED", "FAILED", "PENDING"],
            value="All Sky Status",
            layout=widgets.Layout(width="150px"),
        )

        wandb_filter = widgets.Dropdown(
            options=["All W&B Status", "running", "finished", "failed", "crashed"],
            value="All W&B Status",
            layout=widgets.Layout(width="150px"),
        )

        # Track run controls
        track_label = widgets.Label("Track new run:")
        track_input = widgets.Textarea(
            value="",
            placeholder="Enter run ID",
            layout=widgets.Layout(width="200px", height="30px"),
            style={"description_width": "initial"},
            continuous_update=False,  # Only update on Enter or blur
            rows=1,
        )

        track_button = widgets.Button(
            description="Track Run", button_style="primary", layout=widgets.Layout(width="100px")
        )

        refresh_all_button = widgets.Button(
            description="Refresh All", button_style="success", layout=widgets.Layout(width="100px")
        )

        refresh_table_button = widgets.Button(
            description="Refresh Table", button_style="info", layout=widgets.Layout(width="100px")
        )

        # Output area for the table
        output = widgets.HTML()

        # Pagination controls
        prev_button = widgets.Button(description="Previous", disabled=True, layout=widgets.Layout(width="80px"))

        next_button = widgets.Button(description="Next", disabled=True, layout=widgets.Layout(width="80px"))

        page_label = widgets.Label(value="Page 1")

        # Status label
        status_label = widgets.Label(value=f"Showing {len(runs)} runs")

        def update_table():
            """Update the table display based on current filters and sorting."""
            nonlocal runs
            runs = self.get_all()  # Refresh data

            # Apply filters
            filtered = runs

            # Search filter
            search_term = search_input.value.strip().lower()
            if search_term:
                filtered = [r for r in filtered if search_term in r.run_id.lower()]

            # Sky filter
            if sky_filter.value != "All Sky Status":
                filtered = [r for r in filtered if r.sky and r.sky.status.value == sky_filter.value]

            # W&B filter
            if wandb_filter.value != "All W&B Status":
                filtered = [r for r in filtered if r.wandb and r.wandb.status.value == wandb_filter.value]

            self._filtered_runs = filtered

            # Sort
            if self._sort_column == "created":
                self._filtered_runs.sort(key=lambda r: r.created_at, reverse=not self._sort_ascending)
            elif self._sort_column == "run_id":
                self._filtered_runs.sort(key=lambda r: r.run_id, reverse=not self._sort_ascending)

            # Pagination
            total_pages = max(1, (len(self._filtered_runs) + max_rows - 1) // max_rows)
            self._current_page = min(self._current_page, total_pages - 1)

            start_idx = self._current_page * max_rows
            end_idx = min(start_idx + max_rows, len(self._filtered_runs))
            page_runs = self._filtered_runs[start_idx:end_idx]

            # Update pagination controls
            prev_button.disabled = self._current_page == 0
            next_button.disabled = self._current_page >= total_pages - 1
            page_label.value = f"Page {self._current_page + 1} of {total_pages}"
            status_label.value = f"Showing {start_idx + 1}-{end_idx} of {len(self._filtered_runs)} runs"

            # Generate HTML table
            if not page_runs:
                output.value = '<p style="text-align: center; color: #666;">No runs match the filters</p>'
                return

            # Create HTML table
            html = """
            <style>
            .runstore-table {
                width: 100%;
                border-collapse: collapse;
                font-family: monospace;
                font-size: 13px;
            }
            .runstore-table th, .runstore-table td {
                padding: 8px 12px;
                text-align: right;
                border-bottom: 1px solid #e0e0e0;
            }
            .runstore-table th {
                font-weight: bold;
                background: #f8f9fa;
                border-bottom: 2px solid #e0e0e0;
            }
            .runstore-table tr:hover {
                background: #f8f9fa;
            }
            .status-badge {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
                font-weight: 600;
            }
            .status-running { background: #e3f2fd; color: #1976d2; }
            .status-pending { background: #fff3cd; color: #856404; }
            .status-failed { background: #fee; color: #c62828; }
            .status-completed, .status-succeeded { background: #e8f5e9; color: #2e7d32; }
            .status-unsubmitted { background: #f5f5f5; color: #757575; }
            </style>
            <table class="runstore-table">
            <thead>
                <tr>
                    <th>Run ID</th>
                    <th>SkyPilot</th>
                    <th>W&B</th>
                    <th>Created</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
            """

            for idx, run in enumerate(page_runs):
                # Format values
                if run.sky:
                    sky_status_class = f"status-{run.sky.status.value.lower()}"
                    sky_str = f'<span class="status-badge {sky_status_class}">{run.sky.status.value}</span>'
                else:
                    sky_str = "—"

                if run.wandb:
                    wandb_status_class = f"status-{run.wandb.status.value.lower()}"
                    wandb_str = (
                        f'<a href="{run.wandb.url}" target="_blank"><span class="status-badge {wandb_status_class}">'
                        f"{run.wandb.status.value}</span></a>"
                    )
                else:
                    wandb_str = "—"

                created_str = run.created_at.strftime("%Y-%m-%d %H:%M:%S")

                html += f"""
                <tr>
                    <td>{run.run_id}</td>
                    <td>{sky_str}</td>
                    <td>{wandb_str}</td>
                    <td>{created_str}</td>
                    <td>
                    <button id="refresh-btn-{idx}" onclick="refreshRun{idx}()"style="padding:4px 8px;font-size:11px;">
                        Refresh
                    </button>
                    </td>
                </tr>
                """

            html += "</tbody></table>"

            # Add JavaScript for individual refresh buttons
            html += f"""
            <script>
            (function() {{
                // Function to refresh individual run
                function refreshRun(runId, buttonId) {{
                    console.log('Refreshing run:', runId);

                    // Find the specific button and update its text
                    const button = document.getElementById(buttonId);
                    if (button) {{
                        button.disabled = true;
                        button.textContent = 'Refreshing...';
                    }}

                    // Use Jupyter's kernel to execute Python code
                    if (typeof Jupyter !== 'undefined' && Jupyter.notebook && Jupyter.notebook.kernel) {{
                        const startTime = Date.now();

                        Jupyter.notebook.kernel.execute(
                            `# Refresh run in RunStore\\n` +
                            `from notebooks.run_store import get_runstore\\n` +
                            `rs = get_runstore()\\n` +
                            `updated = rs.refresh_run('${{runId}}')\\n` +
                            `print(f'Refreshed ${{runId}}: Updated={{updated}}')`,
                            {{
                                iopub: {{
                                    output: function(msg) {{
                                        if (msg.msg_type === 'stream') {{
                                            console.log('Refresh output:', msg.content.text);
                                        }}
                                    }}
                                }},
                                shell: {{
                                    reply: function(msg) {{
                                        // Command completed
                                        const elapsed = Date.now() - startTime;
                                        console.log(`Refresh completed in ${{elapsed}}ms`);

                                        // Update button state
                                        if (button) {{
                                            button.textContent = 'Refreshed!';
                                            setTimeout(() => {{
                                                button.disabled = false;
                                                button.textContent = 'Refresh';
                                            }}, 1500);
                                        }}
                                    }}
                                }}
                            }}
                        );
                    }} else {{
                        console.error('Jupyter kernel not available');
                        if (button) {{
                            button.disabled = false;
                            button.textContent = 'Error';
                            setTimeout(() => {{
                                button.textContent = 'Refresh';
                            }}, 1500);
                        }}
                    }}
                }}

                // Make refreshRun functions available globally with unique names
                // Bind refresh functions
                {
                "; ".join(
                    [
                        f'window.refreshRun{idx} = () => refreshRun("{run.run_id}", "refresh-btn-{idx}")'
                        for idx, run in enumerate(page_runs)
                    ]
                )
            }
            }})();
            </script>
            """

            output.value = html

        # Event handlers
        def on_search_change(change):
            self._current_page = 0
            update_table()

        def on_filter_change(change):
            self._current_page = 0
            update_table()

        def on_track_click(b):
            run_id = track_input.value.strip()
            if run_id:
                self.add_run(run_id)
                track_input.value = ""
                update_table()

        def on_refresh_all_click(b):
            b.disabled = True
            b.description = "Refreshing..."
            try:
                self.refresh_all()
                update_table()
            finally:
                b.disabled = False
                b.description = "Refresh All"

        def on_refresh_table_click(b):
            update_table()
            # Brief visual feedback
            original_style = b.button_style
            b.button_style = "success"
            b.description = "Refreshed!"
            import threading

            def reset():
                b.button_style = original_style
                b.description = "Refresh Table"

            threading.Timer(0.5, reset).start()

        def on_prev_click(b):
            self._current_page = max(0, self._current_page - 1)
            update_table()

        def on_next_click(b):
            self._current_page += 1
            update_table()

        # Connect event handlers
        search_input.observe(on_search_change, "value")
        sky_filter.observe(on_filter_change, "value")
        wandb_filter.observe(on_filter_change, "value")
        track_button.on_click(on_track_click)
        # Note: Textarea doesn't support on_submit, so users need to click the button
        refresh_all_button.on_click(on_refresh_all_click)
        refresh_table_button.on_click(on_refresh_table_click)
        prev_button.on_click(on_prev_click)
        next_button.on_click(on_next_click)

        # No auto-refresh - keep it simple

        # Layout
        filters_row = widgets.HBox(
            [search_label, search_input, sky_filter, wandb_filter],
            layout=widgets.Layout(gap="10px", align_items="center"),
        )

        track_row = widgets.HBox(
            [track_label, track_input, track_button, refresh_all_button, refresh_table_button],
            layout=widgets.Layout(gap="10px", align_items="center"),
        )

        header_section = widgets.VBox([filters_row, track_row], layout=widgets.Layout(gap="10px", margin="0 0 10px 0"))

        pagination_row = widgets.HBox(
            [status_label, widgets.HBox([prev_button, page_label, next_button], layout=widgets.Layout(gap="5px"))],
            layout=widgets.Layout(justify_content="space-between", margin="10px 0"),
        )

        # Main container
        container = widgets.VBox([header_section, output, pagination_row])

        # No cleanup needed without auto-refresh

        # Store reference to update function for manual refresh
        container.update_table = update_table

        # Initial render
        update_table()

        return container

    def get_active_runs(self) -> list[Run]:
        return [run for run in self.get_all() if run.is_active]

    def refresh_active_runs(self) -> list[Run]:
        """Refresh only runs that are still active (non-terminated)."""
        # Load current data
        store_data = self._load()

        # Get active run IDs
        active_run_ids = [run.run_id for run in store_data.runs.values() if run.is_active]

        if not active_run_ids:
            return []

        # Batch fetch from both sources
        sky_jobs = self._sky_client.get_jobs_by_names(active_run_ids)
        wandb_runs = self._wandb_client.get_runs_by_ids(active_run_ids)

        updated_runs = []

        # Update each run with fresh data
        for run_id in active_run_ids:
            run = store_data.runs[run_id]
            updated = False

            if run_id in sky_jobs:
                run.sky = sky_jobs[run_id]
                updated = True

            if run_id in wandb_runs:
                run.wandb = wandb_runs[run_id]
                updated = True

            if updated:
                run.updated_at = datetime.now()
                updated_runs.append(run)

        if updated_runs:
            self._save(store_data)

        return updated_runs

    def get_unsubmitted_runs(self) -> list[Run]:
        return [run for run in self.get_all() if not run.sky and not run.wandb]

    def discover_wandb_runs(self, project: str = "metta", entity: str = "metta-research", limit: int = 20) -> list[str]:
        store_data = self._load()
        exclude_ids = set(store_data.runs.keys())
        return self._wandb_client.list_all(project, entity, limit, exclude_ids)

    def discover_sky_jobs(self) -> list[str]:
        store_data = self._load()
        exclude_names = set(store_data.runs.keys())
        return self._sky_client.discover_jobs(exclude_names)

import json
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from notebooks.clients.sky_client import SkyPilotClient, SkyPilotJobData, SkyStatus
from notebooks.clients.wandb_client import RunConfig, WandBClient, WandBRunData, WandBStatus


class RunStatus(str, Enum):
    UNSUBMITTED = "unsubmitted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    SUBMITTED = "submitted"


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

    @property
    def status(self) -> RunStatus:
        if not self.sky and not self.wandb:
            return RunStatus.UNSUBMITTED if self.config.command_args else RunStatus.UNKNOWN

        if self.sky:
            if self.sky.status == SkyStatus.PENDING:
                return RunStatus.PENDING
            elif self.sky.status == SkyStatus.RUNNING:
                return RunStatus.RUNNING
            elif self.sky.status == SkyStatus.SUCCEEDED:
                return RunStatus.COMPLETED
            elif self.sky.status == SkyStatus.FAILED:
                return RunStatus.FAILED

        if self.wandb:
            if self.wandb.status == WandBStatus.RUNNING:
                return RunStatus.RUNNING
            elif self.wandb.status == WandBStatus.FINISHED:
                return RunStatus.COMPLETED
            elif self.wandb.status in (WandBStatus.FAILED, WandBStatus.CRASHED):
                return RunStatus.FAILED

        return RunStatus.SUBMITTED


class RunStoreData(BaseModel):
    runs: dict[str, Run] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)


class RunStore:
    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".metta" / "run_store.json"

        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._store_data = RunStoreData()
        self._load()

        self._sky_client = SkyPilotClient()
        self._wandb_client = WandBClient()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self._store_data = RunStoreData.model_validate(data)
            except Exception as e:
                print(f"Warning: Could not load run store: {e}")
                self._store_data = RunStoreData()

    def _save(self) -> None:
        with self._lock:
            if self.storage_path.exists():
                backup_path = self.storage_path.with_suffix(".json.bak")
                self.storage_path.rename(backup_path)

            self._store_data.updated_at = datetime.now()

            data = self._store_data.model_dump(mode="json")

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def add_run(self, run_id: str, command_args: str | None = None) -> Run:
        run_id = run_id.strip()
        print(f"[RunStore] Adding run: {run_id}")

        if run_id in self._store_data.runs:
            print(f"[RunStore] Run '{run_id}' already exists, updating...")
            run = self._store_data.runs[run_id]
            if command_args:
                run.config.command_args = command_args
                run.updated_at = datetime.now()
        else:
            print(f"[RunStore] Creating new run: {run_id}")
            config = RunConfig()
            if command_args:
                config.command_args = command_args

            run = Run(run_id=run_id, config=config)
            self._store_data.runs[run_id] = run

        self._save()
        print(f"[RunStore] Saved to {self.storage_path}")

        # Verify it was saved
        print(f"[RunStore] Total runs in store: {len(self._store_data.runs)}")

        # Auto-update run status after adding
        # self.update(run_id)

        return self._store_data.runs[run_id]

    def get_all(self) -> list[Run]:
        runs = list(self._store_data.runs.values())
        runs.sort(key=lambda r: r.updated_at, reverse=True)
        return runs

    def get_run(self, run_id: str) -> Run | None:
        return self._store_data.runs.get(run_id)

    def remove_run(self, run_id: str) -> bool:
        if run_id in self._store_data.runs:
            del self._store_data.runs[run_id]
            self._save()
            return True
        return False

    def load_in_all(self) -> None:
        sky_runs = self._sky_client.get_all()
        wandb_runs = {run_id: self._wandb_client.get_run(run_id) for run_id in self._wandb_client.list_all()}

        run_ids = set(sky_runs.keys()) | set(wandb_runs.keys())

        for run_id in run_ids:
            if run_id not in self._store_data.runs:
                self._store_data.runs[run_id] = Run(run_id=run_id)
            if run_id in sky_runs:
                self._store_data.runs[run_id].sky = sky_runs[run_id]
            if run_id in wandb_runs:
                self._store_data.runs[run_id].wandb = wandb_runs[run_id]
        self._save()

    def refresh_all(self) -> list[Run]:
        # Get all run IDs
        run_ids = list(self._store_data.runs.keys())

        # Batch fetch from both sources
        sky_jobs = self._sky_client.get_jobs_by_names(run_ids)
        wandb_runs = self._wandb_client.get_runs_by_ids(run_ids)

        updated_runs = []

        # Update each run with fresh data
        for run_id in run_ids:
            run = self._store_data.runs[run_id]
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
            self._save()

        return updated_runs

    def refresh_run(self, run_id: str) -> bool:
        print(f"[RunStore] Refreshing run: {run_id}")

        if run_id not in self._store_data.runs:
            print(f"[RunStore] Run {run_id} not found in store")
            return False

        run = self._store_data.runs[run_id]
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
            self._save()
            print(f"[RunStore] Updated run {run_id} and saved")
        else:
            print(f"[RunStore] No updates found for run {run_id}")

        return updated

    def to_html_table(self, runs: list[Run] | None = None, max_rows: int = 10, table_id: str | None = None) -> str:
        if runs is None:
            runs = self.get_all()

        if not runs:
            return '<div style="padding: 20px; text-align: center; color: #666;">No runs tracked</div>'
        print("Num runs:", len(runs))

        # Generate unique ID for this table instance
        if table_id is None:
            table_id = f"runstore_{id(self)}_{datetime.now().timestamp()}".replace(".", "")

        # JavaScript and CSS for interactive table
        html = f"""
        <style>
        #{table_id}_container {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 13px;
            color: #333;
        }}

        #{table_id}_controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
        }}

        #{table_id}_search {{
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            width: 250px;
            font-size: 13px;
        }}

        #{table_id}_controls select {{
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            font-size: 13px;
            background: white;
        }}

        #{table_id}_track_run {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        #{table_id}_track_input {{
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            width: 200px;
            font-size: 13px;
        }}

        #{table_id}_track_button {{
            padding: 8px 16px;
            border: 1px solid #1976d2;
            background: #1976d2;
            color: white;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 500;
        }}

        #{table_id}_track_button:hover {{
            background: #1565c0;
            border-color: #1565c0;
        }}

        #{table_id}_table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        #{table_id}_table th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: right;
            font-weight: 600;
            color: #555;
            border-bottom: 2px solid #e0e0e0;
            cursor: pointer;
            user-select: none;
        }}

        #{table_id}_table th:hover {{
            background: #f0f1f3;
        }}

        #{table_id}_table td {{
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            text-align: right;
        }}

        #{table_id}_table tr:hover {{
            background: #f8f9fa;
        }}

        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .status-running {{
            background: #e3f2fd;
            color: #1976d2;
        }}

        .status-pending {{
            background: #fff3cd;
            color: #856404;
        }}

        .status-failed {{
            background: #fee;
            color: #c62828;
        }}

        .status-completed, .status-succeeded {{
            background: #e8f5e9;
            color: #2e7d32;
        }}

        .status-unsubmitted {{
            background: #f5f5f5;
            color: #757575;
        }}

        .sky-info, .wandb-info {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .wandb-link {{
            color: #1976d2;
            text-decoration: none;
            font-weight: 500;
        }}

        .wandb-link:hover {{
            text-decoration: underline;
        }}

        #{table_id}_pagination {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            font-size: 13px;
            color: #666;
        }}

        #{table_id}_pagination button {{
            padding: 6px 12px;
            border: 1px solid #e0e0e0;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}

        #{table_id}_pagination button:hover:not(:disabled) {{
            background: #f8f9fa;
        }}

        #{table_id}_pagination button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .sort-arrow {{
            margin-left: 5px;
            font-size: 10px;
            color: #999;
        }}

        .sort-arrow.active {{
            color: #333;
        }}
        
        #{table_id}_refresh_all {{
            padding: 8px 16px;
            border: 1px solid #4caf50;
            background: #4caf50;
            color: white;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 500;
            margin-left: 10px;
        }}
        
        #{table_id}_refresh_all:hover {{
            background: #45a049;
            border-color: #45a049;
        }}
        
        #{table_id}_refresh_all:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}
        
        .refresh-row-btn {{
            padding: 4px 8px;
            border: 1px solid #e0e0e0;
            background: white;
            color: #333;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            margin-left: 8px;
        }}
        
        .refresh-row-btn:hover {{
            background: #f8f9fa;
            border-color: #ccc;
        }}
        
        .refresh-row-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        #{table_id}_loading {{
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px 30px;
            border-radius: 8px;
            z-index: 9999;
            text-align: center;
        }}
        
        #{table_id}_loading.active {{
            display: block;
        }}
        
        .loading-spinner {{
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>

        <div id="{table_id}_container">
            <div id="{table_id}_controls">
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <input type="text" id="{table_id}_search" placeholder="Search run ID..." />

                    <select id="{table_id}_filter_status">
                        <option value="">All Statuses</option>
                        <option value="running">Running</option>
                        <option value="completed">Completed</option>
                        <option value="failed">Failed</option>
                        <option value="pending">Pending</option>
                        <option value="unsubmitted">Unsubmitted</option>
                    </select>

                    <select id="{table_id}_filter_sky">
                        <option value="">All Sky Status</option>
                        <option value="RUNNING">Sky Running</option>
                        <option value="SUCCEEDED">Sky Succeeded</option>
                        <option value="FAILED">Sky Failed</option>
                        <option value="PENDING">Sky Pending</option>
                    </select>

                    <select id="{table_id}_filter_wandb">
                        <option value="">All W&B Status</option>
                        <option value="running">W&B Running</option>
                        <option value="finished">W&B Finished</option>
                        <option value="failed">W&B Failed</option>
                        <option value="crashed">W&B Crashed</option>
                    </select>
                </div>
                
                <div style="display: flex; gap: 10px; align-items: center;">
                    <div id="{table_id}_track_run">
                        <input type="text" id="{table_id}_track_input" placeholder="Run ID to track" />
                        <button id="{table_id}_track_button">Track Run</button>
                    </div>
                    <button id="{table_id}_refresh_all">Refresh All</button>
                </div>
            </div>

            <table id="{table_id}_table">
                <thead>
                    <tr>
                        <th data-sort="run_id">Run ID <span class="sort-arrow">▼</span></th>
                        <th data-sort="status">Status <span class="sort-arrow"></span></th>
                        <th data-sort="sky">SkyPilot <span class="sort-arrow"></span></th>
                        <th data-sort="wandb">W&B <span class="sort-arrow"></span></th>
                        <th data-sort="created">Created <span class="sort-arrow"></span></th>
                        <th style="width: 80px;">Actions</th>
                    </tr>
                </thead>
                <tbody id="{table_id}_tbody">
                </tbody>
            </table>

            <div id="{table_id}_pagination">
                <div>
                    Showing <span id="{table_id}_showing">0</span> of <span id="{table_id}_total">0</span> runs
                </div>
                <div>
                    <button id="{table_id}_prev" onclick="{table_id}_changePage(-1)">Previous</button>
                    <span id="{table_id}_page_info" style="margin: 0 10px;">Page 1</span>
                    <button id="{table_id}_next" onclick="{table_id}_changePage(1)">Next</button>
                </div>
            </div>
            
            <div id="{table_id}_loading">
                <div class="loading-spinner"></div>
                <div>Refreshing runs...</div>
            </div>
        </div>

        <script>
        (function() {{
            // Debug: Log when script starts
            console.log('RunStore table script starting...');
        """

        # Add run data as JavaScript array
        run_data = []
        for run in runs:
            try:
                # Get created timestamp - prefer W&B created_at if available
                created_at = run.created_at
                if run.wandb and run.wandb.created_at:
                    created_at = run.wandb.created_at

                # Format timestamp
                if isinstance(created_at, str):
                    created_str = created_at
                elif hasattr(created_at, "isoformat"):
                    created_str = created_at.isoformat()
                else:
                    created_str = str(created_at)

                # Ensure status has value attribute
                status_value = run.status.value if hasattr(run.status, "value") else str(run.status).lower()

                run_obj = {
                    "run_id": run.run_id,
                    "status": status_value,
                    "sky_id": run.sky.job_id if run.sky else "",
                    "sky_status": run.sky.status.value if run.sky and hasattr(run.sky.status, "value") else "",
                    "wandb_url": run.wandb.url if run.wandb else "",
                    "wandb_status": run.wandb.status.value if run.wandb and hasattr(run.wandb.status, "value") else "",
                    "created": created_str,
                }
                run_data.append(run_obj)
            except Exception as e:
                print(f"Error processing run {run.run_id}: {e}")
                # Add a minimal entry for debugging
                run_data.append(
                    {
                        "run_id": str(run.run_id),
                        "status": "error",
                        "sky_id": "",
                        "sky_status": "",
                        "wandb_url": "",
                        "wandb_status": "",
                        "created": datetime.now().isoformat(),
                    }
                )
        print("numrun data", len(run_data))

        import json

        html += "const data = " + json.dumps(run_data) + ";"

        html += f"""

            // Debug: Log data
            console.log('RunStore data loaded:', data.length, 'runs');
            if (data.length > 0) {{
                console.log('First run:', data[0]);
                console.log('Sample created dates:', data.slice(0, 3).map(r => r.created));
            }}

            let currentPage = 0;
            const rowsPerPage = {max_rows};
            let filteredData = [...data];
            let sortColumn = 'created';
            let sortAsc = false;

            function updateTable() {{
                console.log('updateTable called, filteredData length:', filteredData.length);
                const tbody = document.getElementById('{table_id}_tbody');
                tbody.innerHTML = '';

                // Apply filters
                filteredData = data.filter(run => {{
                    const searchTerm = document.getElementById('{table_id}_search').value.toLowerCase();
                    const statusFilter = document.getElementById('{table_id}_filter_status').value;
                    const skyFilter = document.getElementById('{table_id}_filter_sky').value;
                    const wandbFilter = document.getElementById('{table_id}_filter_wandb').value;

                    if (searchTerm && !run.run_id.toLowerCase().includes(searchTerm)) return false;
                    if (statusFilter && run.status !== statusFilter) return false;
                    if (skyFilter && run.sky_status !== skyFilter) return false;
                    if (wandbFilter && run.wandb_status !== wandbFilter) return false;

                    return true;
                }});

                // Sort data
                filteredData.sort((a, b) => {{
                    let aVal = a[sortColumn] || '';
                    let bVal = b[sortColumn] || '';

                    if (sortColumn === 'created') {{
                        // Handle date parsing more robustly
                        const aDate = new Date(aVal);
                        const bDate = new Date(bVal);
                        aVal = isNaN(aDate.getTime()) ? 0 : aDate.getTime();
                        bVal = isNaN(bDate.getTime()) ? 0 : bDate.getTime();
                    }}

                    if (aVal < bVal) return sortAsc ? -1 : 1;
                    if (aVal > bVal) return sortAsc ? 1 : -1;
                    return 0;
                }});

                // Paginate
                const start = currentPage * rowsPerPage;
                const end = Math.min(start + rowsPerPage, filteredData.length);
                const pageData = filteredData.slice(start, end);

                // Render rows
                pageData.forEach(run => {{
                    const tr = document.createElement('tr');

                    // Run ID
                    const tdRunId = document.createElement('td');
                    tdRunId.textContent = run.run_id;
                    tr.appendChild(tdRunId);

                    // Status
                    const tdStatus = document.createElement('td');
                    tdStatus.innerHTML = `<span class="status-badge status-${{run.status}}">${{run.status}}</span>`;
                    tr.appendChild(tdStatus);

                    // SkyPilot
                    const tdSky = document.createElement('td');
                    if (run.sky_id) {{
                        tdSky.innerHTML = `
                            <div class="sky-info">
                                <span>${{run.sky_id}}</span>
                                <span class="status-badge status-${{run.sky_status.toLowerCase()}}">
                                    ${{run.sky_status}}
                                </span>
                            </div>
                        `;
                    }} else {{
                        tdSky.innerHTML = '<span style="color: #999;">—</span>';
                    }}
                    tr.appendChild(tdSky);

                    // W&B
                    const tdWandb = document.createElement('td');
                    if (run.wandb_url) {{
                        tdWandb.innerHTML = `
                            <a href="${{run.wandb_url}}" target="_blank" class="wandb-link">
                                <span class="status-badge status-${{run.wandb_status}}">${{run.wandb_status}}</span>
                            </a>
                        `;
                    }} else {{
                        tdWandb.innerHTML = '<span style="color: #999;">—</span>';
                    }}
                    tr.appendChild(tdWandb);

                    // Created
                    const tdCreated = document.createElement('td');
                    const date = new Date(run.created);
                    tdCreated.textContent = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                    tr.appendChild(tdCreated);
                    
                    // Actions
                    const tdActions = document.createElement('td');
                    tdActions.innerHTML = `
                        <button class="refresh-row-btn" onclick="{table_id}_refreshRun('${{run.run_id}}')">
                            Refresh
                        </button>
                    `;
                    tr.appendChild(tdActions);

                    tbody.appendChild(tr);
                }});

                // Update pagination
                document.getElementById('{table_id}_showing').textContent =
                    filteredData.length === 0 ? 0 : `${{start + 1}}-${{end}}`;
                document.getElementById('{table_id}_total').textContent = filteredData.length;
                document.getElementById('{table_id}_page_info').textContent =
                    `Page ${{currentPage + 1}} of ${{Math.ceil(filteredData.length / rowsPerPage) || 1}}`;

                document.getElementById('{table_id}_prev').disabled = currentPage === 0;
                document.getElementById('{table_id}_next').disabled =
                    end >= filteredData.length;
            }}

            window.{table_id}_changePage = function(delta) {{
                currentPage = Math.max(0, Math.min(
                    currentPage + delta,
                    Math.ceil(filteredData.length / rowsPerPage) - 1
                ));
                updateTable();
            }};

            // Sort handlers
            document.querySelectorAll('#{table_id}_table th[data-sort]').forEach(th => {{
                th.addEventListener('click', function() {{
                    const column = this.getAttribute('data-sort');

                    // Update sort arrows
                    document.querySelectorAll('#{table_id}_table .sort-arrow').forEach(arrow => {{
                        arrow.classList.remove('active');
                        arrow.textContent = '';
                    }});

                    if (sortColumn === column) {{
                        sortAsc = !sortAsc;
                    }} else {{
                        sortColumn = column;
                        sortAsc = true;
                    }}

                    this.querySelector('.sort-arrow').classList.add('active');
                    this.querySelector('.sort-arrow').textContent = sortAsc ? '▲' : '▼';

                    currentPage = 0;
                    updateTable();
                }});
            }});

            // Filter handlers
            document.getElementById('{table_id}_search').addEventListener('input', () => {{
                currentPage = 0;
                updateTable();
            }});

            document.getElementById('{table_id}_filter_status').addEventListener('change', () => {{
                currentPage = 0;
                updateTable();
            }});

            document.getElementById('{table_id}_filter_sky').addEventListener('change', () => {{
                currentPage = 0;
                updateTable();
            }});

            document.getElementById('{table_id}_filter_wandb').addEventListener('change', () => {{
                currentPage = 0;
                updateTable();
            }});
            
            // Function to add a new run
            function addNewRun() {{
                const runId = document.getElementById('{table_id}_track_input').value.trim();
                if (runId) {{
                    console.log('Adding new run:', runId);
                    
                    // Check if run already exists
                    const existingRun = data.find(r => r.run_id === runId);
                    if (existingRun) {{
                        alert(`Run "${{runId}}" is already being tracked.`);
                        return;
                    }}
                    
                    // Add the run to the data immediately
                    const newRun = {{
                        run_id: runId,
                        status: 'pending',
                        sky_id: '',
                        sky_status: '',
                        wandb_url: '',
                        wandb_status: '',
                        created: new Date().toISOString()
                    }};
                    
                    // Add to data array
                    data.push(newRun);
                    
                    // Clear input
                    document.getElementById('{table_id}_track_input').value = '';
                    
                    // Re-render table
                    currentPage = 0; // Reset to first page
                    updateTable();
                    
                    // If there's a callback to actually add to RunStore, call it
                    if (window.{table_id}_track_run_callback) {{
                        window.{table_id}_track_run_callback(runId);
                    }} else {{
                        console.log('No track_run_callback defined. Run added to table only.');
                    }}
                }}
            }}
            
            // Track Run button handler
            document.getElementById('{table_id}_track_button').addEventListener('click', addNewRun);
            
            // Handle Enter key in track input
            document.getElementById('{table_id}_track_input').addEventListener('keypress', (e) => {{
                if (e.key === 'Enter') {{
                    addNewRun();
                }}
            }});
            
            // Refresh All button handler
            document.getElementById('{table_id}_refresh_all').addEventListener('click', async () => {{
                console.log('Refresh All clicked');
                document.getElementById('{table_id}_loading').classList.add('active');
                document.getElementById('{table_id}_refresh_all').disabled = true;
                
                // Disable all row refresh buttons
                document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = true);
                
                // Call parent refresh_all callback if available
                if (window.{table_id}_refresh_all_callback) {{
                    try {{
                        await window.{table_id}_refresh_all_callback();
                    }} catch (e) {{
                        console.error('Error refreshing all:', e);
                    }}
                }} else {{
                    // Simulate refresh for demo
                    setTimeout(() => {{
                        document.getElementById('{table_id}_loading').classList.remove('active');
                        document.getElementById('{table_id}_refresh_all').disabled = false;
                        document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = false);
                        alert('To enable refresh, integrate with RunStore.refresh_all()');
                    }}, 1000);
                }}
            }});
            
            // Refresh single run function
            window.{table_id}_refreshRun = async function(runId) {{
                console.log('Refresh run:', runId);
                document.getElementById('{table_id}_loading').classList.add('active');
                
                // Disable all buttons during refresh
                document.getElementById('{table_id}_refresh_all').disabled = true;
                document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = true);
                
                // Call parent refresh_run callback if available
                if (window.{table_id}_refresh_run_callback) {{
                    try {{
                        await window.{table_id}_refresh_run_callback(runId);
                    }} catch (e) {{
                        console.error('Error refreshing run:', e);
                    }}
                }} else {{
                    // Simulate refresh for demo
                    setTimeout(() => {{
                        document.getElementById('{table_id}_loading').classList.remove('active');
                        document.getElementById('{table_id}_refresh_all').disabled = false;
                        document.querySelectorAll('.refresh-row-btn').forEach(btn => btn.disabled = false);
                        alert('To enable refresh, integrate with RunStore.refresh_run("' + runId + '")');
                    }}, 1000);
                }}
            }};
            
            // Function to update data and re-render
            window.{table_id}_updateData = function(newData) {{
                data.length = 0;
                data.push(...newData);
                updateTable();
            }};

            // Initial render
            updateTable();

            // Set initial sort on created column
            document.querySelector('#{table_id}_table th[data-sort="created"]').click();
        }})();
        </script>
        """

        # Simple Jupyter integration - just re-run the cell to refresh
        if table_id and table_id.startswith("runstore_"):
            integration_script = f"""
            <script>
            // Simple integration - refresh by re-running the cell
            window.{table_id}_refresh_all_callback = function() {{
                alert('Re-run this cell to refresh the table');
            }};
            window.{table_id}_refresh_run_callback = function(runId) {{
                alert('Re-run this cell to refresh the table');
            }};
            window.{table_id}_track_run_callback = function(runId) {{
                alert('Use add_test_run("' + runId + '") then re-run this cell');
            }};
            </script>
            """
            return html + integration_script

        return html

    def get_active_runs(self) -> list[Run]:
        return [run for run in self.get_all() if run.is_active]

    def refresh_active_runs(self) -> list[Run]:
        """Refresh only runs that are still active (non-terminated)."""
        # Get active run IDs
        active_run_ids = [run.run_id for run in self.get_all() if run.is_active]

        if not active_run_ids:
            return []

        # Batch fetch from both sources
        sky_jobs = self._sky_client.get_jobs_by_names(active_run_ids)
        wandb_runs = self._wandb_client.get_runs_by_ids(active_run_ids)

        updated_runs = []

        # Update each run with fresh data
        for run_id in active_run_ids:
            run = self._store_data.runs[run_id]
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
            self._save()

        return updated_runs

    def get_unsubmitted_runs(self) -> list[Run]:
        return [run for run in self.get_all() if run.status == RunStatus.UNSUBMITTED]

    def discover_wandb_runs(self, project: str = "metta", entity: str = "metta-research", limit: int = 20) -> list[str]:
        exclude_ids = set(self._store_data.runs.keys())
        return self._wandb_client.list_all(project, entity, limit, exclude_ids)

    def discover_sky_jobs(self) -> list[str]:
        exclude_names = set(self._store_data.runs.keys())
        return self._sky_client.discover_jobs(exclude_names)

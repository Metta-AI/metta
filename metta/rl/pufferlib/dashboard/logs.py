import os

from rich.table import Table

from .dashboard import ROUND_OPEN, DashboardComponent, c1, c2

class Logs(DashboardComponent):
    def __init__(self, logs_path: str, max_lines=5):
        super().__init__()
        self.logs_path = logs_path
        self.stdout_log_path = os.path.join(logs_path, "out.log")
        self.stderr_log_path = os.path.join(logs_path, "error.log")
        self.max_lines = max_lines

    def update(self):
        pass

    def render(self):
        stdout = Table(box=ROUND_OPEN, expand=False, pad_edge=False)
        stderr = Table(box=ROUND_OPEN, expand=False, pad_edge=False)
        with open(self.stdout_log_path, "r") as file:
            stdout_log_content = file.readlines()[-self.max_lines :]
            for line in stdout_log_content:
                stdout.add_row(f"{c2}{line.strip()}")
        with open(self.stderr_log_path, "r") as file:
            stderr_log_content = file.readlines()[-self.max_lines :]
            for line in stderr_log_content:
                stderr.add_row(f"{c2}{line.strip()}")
        headers = []
        columns = []
        headers.append(c1 + f"Logs: {self.stdout_log_path}")
        columns.append(stdout)
        if len(stderr_log_content) > 0:
            headers.append(c1 + f"Error: {self.stderr_log_path}")
            columns.append(stderr)

        table = Table(box=ROUND_OPEN, expand=True, show_header=False, border_style="bright_cyan")
        table.add_row(*headers)
        table.add_row(*columns)
        return table

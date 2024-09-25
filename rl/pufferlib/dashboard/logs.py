
import numpy as np
from rich.table import Table


from .dashboard import DashboardComponent
from .dashboard import c1, b1, c2, b2, c3, ROUND_OPEN

class Logs(DashboardComponent):
    def __init__(self, stdout_log_path, stderror_log_path, max_lines=5):
        super().__init__()
        self.stdout_log_path = stdout_log_path
        self.stderror_log_path = stderror_log_path
        self.max_lines = max_lines

    def update(self):
        pass

    def render(self):
        stdout = Table(box=ROUND_OPEN, expand=False, pad_edge=False)
        stderr = Table(box=ROUND_OPEN, expand=False, pad_edge=False)
        with open(self.stdout_log_path, 'r') as file:
            stdout_log_content = file.readlines()[-self.max_lines:]
            for line in stdout_log_content:
                stdout.add_row(f'{c2}{line.strip()}')
        with open(self.stderror_log_path, 'r') as file:
            stderr_log_content = file.readlines()[-self.max_lines:]
            for line in stderr_log_content:
                stderr.add_row(f'{c2}{line.strip()}')
        headers = []
        columns = []
        if len(stdout_log_content) > 0:
            headers.append(c1 + f"Logs: {self.stdout_log_path}")
            columns.append(stdout)
        if len(stderr_log_content) > 0:
            headers.append(c1 + f"Error: {self.stderror_log_path}")
            columns.append(stderr)

        table = Table(box=ROUND_OPEN, expand=True, show_header=False, border_style='bright_cyan')
        table.add_row(*headers)
        table.add_row(*columns)
        return table

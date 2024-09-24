import time
from threading import Thread
import os
from omegaconf import OmegaConf

import rich
from rich.console import Console
from rich.table import Table
from rich.live import Live

class Dashboard(Thread):
    def __init__(self, cfg: OmegaConf, delay=1, components=None):
        super().__init__()
        self.cfg = cfg
        self.delay = delay

        self._components = components or []

        tty_file = open('/dev/tty', 'w')
        self.console = Console(file=tty_file, force_terminal=True)

        self._stopped = False
        self.start()

    def _render_component(self, component, container):
        if isinstance(component, DashboardComponent):
            container.add_row(component.render())
        elif isinstance(component, list):
            container.add_row(*[self.render_component(c, container) for c in component])
        else:
            raise ValueError(f"Invalid component type: {type(component)}")

    def _clear_console(self):
        # Clear console for different operating systems
        if os.name == 'nt':  # For Windows
            os.system('cls')
        else:  # For Unix/Linux/macOS
            os.system('clear')

    def run(self):
        start_time = time.time()
        cleared = False
        with Live(self._render(),
                  console=self.console,
                  redirect_stderr=False,
                  redirect_stdout=False,
                  refresh_per_second=1/self.delay) as live:
            while not self._stopped:
                if time.time() - start_time > 5 and not cleared:
                    self._clear_console()
                    cleared = True
                live.update(self._render())
                time.sleep(self.delay)

    def stop(self):
        for component in self._components:
            component.stop()
        self._stopped = True

    def _render(self):
        dashboard = Table(box=ROUND_OPEN, expand=True,
            show_header=False, border_style='bright_cyan')

        for component in self._components:
            self._render_component(component, dashboard)

        return dashboard

class DashboardComponent:
    def __init__(self):
        pass

    def render(self):
        pass

    def stop(self):
        pass


ROUND_OPEN = rich.box.Box(
    "╭──╮\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = '[bright_cyan]'
c2 = '[white]'
c3 = '[cyan]'
b1 = '[bright_cyan]'
b2 = '[bright_white]'

def abbreviate(num):
    if num < 1e3:
        return f'{b2}{num:.0f}'
    elif num < 1e6:
        return f'{b2}{num/1e3:.1f}{c2}k'
    elif num < 1e9:
        return f'{b2}{num/1e6:.1f}{c2}m'
    elif num < 1e12:
        return f'{b2}{num/1e9:.1f}{c2}b'
    else:
        return f'{b2}{num/1e12:.1f}{c2}t'

def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"

def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100*time/uptime - 1e-5)
    return f'{c1}{name}', duration(time), f'{b2}{percent:2d}%'
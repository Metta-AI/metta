import sys
import termios
from contextlib import contextmanager


@contextmanager
def disable_canonical_mode():
    """Disable canonical mode to allow long token pastes."""
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except (OSError, termios.error):
        yield
        return

    try:
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

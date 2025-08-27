import os
import sys
import time
from contextlib import contextmanager
from typing import Generator

from metta.common.util.text_styles import green, red


@contextmanager
def collapsible_logs(component_name: str) -> Generator[None, None, None]:
    """Final PTY-based collapsible logging - shows output then collapses completely."""

    print(f"  [{component_name}] Installing...", flush=True)

    success = True
    start_time = time.time()

    # Save exact cursor position
    print("\033[s", end="", flush=True)  # Save cursor position

    try:
        # Let ALL output flow completely normally - no PTY complexity
        yield

    except Exception:
        success = False
        raise

    finally:
        duration = time.time() - start_time

        # Give all subprocess output generous time to complete and flush
        # This is crucial for TTY output which can be asynchronous
        # NOTE: this could be buggy, idk.. with async TTY writing who knows.
        time.sleep(1.5)  # Very generous timing for TTY flush

        # Force flush all possible output streams
        sys.stdout.flush()
        sys.stderr.flush()
        os.system("")  # Force terminal flush

        # Now aggressively clear everything back to our saved position
        print("\033[u", end="", flush=True)  # Restore to saved cursor position
        print("\033[0J", end="", flush=True)  # Clear from cursor to end of screen

        # Extra small delay before summary
        time.sleep(0.01)

        # Print clean summary
        status_color = green if success else red
        status_text = "success" if success else "failed"
        status_icon = "✅" if success else "🔴"

        print(
            f"  [{component_name}] install logs collapsed → {status_color(status_text)} {status_icon} ({duration:.1f}s)"
        )

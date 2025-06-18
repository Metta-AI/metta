import argparse
import concurrent.futures
import json
import logging
import os
import signal
import time

import wandb

logger = logging.getLogger(__name__)

# Shared IPC filename, co-located with the heartbeat signal file (must match wandb_context.py)
WANDB_IPC_FILENAME = "wandb_ipc.json"


def record_heartbeat() -> None:
    """Record a heartbeat timestamp to the globally configured file path."""
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE")

    if heartbeat_file_path:
        try:
            # Ensure the directory for the heartbeat file exists
            os.makedirs(os.path.dirname(heartbeat_file_path), exist_ok=True)
            with open(heartbeat_file_path, "w") as f:
                f.write(str(time.time()))
        except Exception as exc:
            logger.warning("Failed to write heartbeat: %s", exc)


# Default timeout for the alert sending operation itself
_ALERT_SEND_TIMEOUT_SECONDS = 30


# Accepts the explicit path to the IPC file
def _send_wandb_alert_with_timeout(title: str, wandb_ipc_file_path: str | None) -> None:
    """Send a W&B alert, reading IPC data from the provided file path."""

    if not wandb_ipc_file_path:
        logger.warning(f"W&B IPC file path not provided. Cannot determine W&B run for alert: '{title}'.")
        return

    # ipc_file_to_read is now wandb_ipc_file_path argument
    run_id_ipc: str | None = None
    project_ipc: str | None = None
    entity_ipc: str | None = None

    try:
        with open(wandb_ipc_file_path, "r") as f:
            ipc_data = json.load(f)
        run_id_ipc = ipc_data.get("run_id")
        project_ipc = ipc_data.get("project")
        entity_ipc = ipc_data.get("entity")

        if not all([run_id_ipc, project_ipc, entity_ipc]):
            missing_keys = [
                key
                for key, val in zip(
                    ["run_id", "project", "entity"], [run_id_ipc, project_ipc, entity_ipc], strict=False
                )
                if not val
            ]
            logger.warning(
                f"Missing required W&B identifiers ({', '.join(missing_keys)}) in IPC file {wandb_ipc_file_path}. "
                f"Cannot send alert for: '{title}'."
            )
            return
    except FileNotFoundError:
        logger.warning(f"W&B IPC file not found at: {wandb_ipc_file_path}. Alert '{title}' not sent.")
        return
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding W&B IPC file {wandb_ipc_file_path}: {e}. Alert '{title}' not sent.")
        return
    except Exception as e:
        logger.error(
            f"Unexpected error reading W&B IPC file {wandb_ipc_file_path}: {e}. Alert '{title}' not sent.",
            exc_info=True,
        )
        return

    def send_alert(alert_title_arg: str, rid: str, proj: str, ent: str) -> None:
        log_ctx, initialized = f"run {ent}/{proj}/{rid}", False
        try:
            wandb.init(
                id=rid,
                project=proj,
                entity=ent,
                resume="must",
                settings=wandb.Settings(init_timeout=15, silent=True, _disable_stats=True, _disable_meta=True),
            )
            initialized = True
            alert_text = "Job terminated due to heartbeat timeout."
            wandb.alert(title=alert_title_arg, text=alert_text)
            logger.info(f"W&B alert '{alert_title_arg}' sent for {log_ctx}. Text: '{alert_text}'")
        except Exception as e:
            is_wandb_specific_error = isinstance(e, wandb.errors.Error)
            (logger.warning if is_wandb_specific_error else logger.error)(
                (
                    f"{'W&B ' if is_wandb_specific_error else 'Unexpected '}error in alert for {log_ctx} "
                    f"(trigger: '{alert_title_arg}'): {type(e).__name__} - {e}"
                ),
                exc_info=not is_wandb_specific_error,
            )
        finally:
            if initialized:
                try:
                    wandb.finish()
                except Exception as finish_exception:
                    logger.warning(f"Error during wandb.finish() for {log_ctx}: {finish_exception}")

    if run_id_ipc and project_ipc and entity_ipc:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(send_alert, title, run_id_ipc, project_ipc, entity_ipc)
            try:
                future.result(timeout=_ALERT_SEND_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                logger.warning(f"W&B alert '{title}' sending timed out after {_ALERT_SEND_TIMEOUT_SECONDS}s.")
            except Exception as e:
                logger.warning(f"Exception during W&B alert '{title}' execution: {type(e).__name__} - {e}")


# monitor_heartbeat derives and passes the wandb_ipc_file_path
def monitor_heartbeat(file_path: str, pid: int, timeout: float = 600.0, check_interval: float = 60.0) -> None:
    """Monitor the heartbeat file and terminate the process group if stale."""

    while True:
        time.sleep(check_interval)
        try:
            last = os.path.getmtime(file_path)
        except FileNotFoundError:
            last = 0.0
        if time.time() - last > timeout:
            logger.error("No heartbeat detected for %s seconds. Terminating job", timeout)

            # Derive IPC file path from the heartbeat signal file path
            wandb_ipc_file_path_derived: str | None = None
            try:
                abs_heartbeat_path = os.path.abspath(file_path)
                ipc_dir = os.path.dirname(abs_heartbeat_path)
                wandb_ipc_file_path_derived = os.path.join(ipc_dir, WANDB_IPC_FILENAME)
            except Exception as e:
                logger.error(
                    f"Error deriving W&B IPC file path from heartbeat signal path '{file_path}': {e}", exc_info=True
                )

            _send_wandb_alert_with_timeout(
                title="Heartbeat Timeout. Job terminated.", wandb_ipc_file_path=wandb_ipc_file_path_derived
            )
            time.sleep(_ALERT_SEND_TIMEOUT_SECONDS)  # Allow time for alert to potentially send before killing

            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(10)
            os.killpg(pid, signal.SIGKILL)
            break


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    hb = sub.add_parser("heartbeat")
    hb.add_argument("file")

    mon = sub.add_parser("monitor")
    mon.add_argument("file")
    mon.add_argument("--pid", type=int, default=os.getpid())
    mon.add_argument("--timeout", type=float, default=600.0)
    mon.add_argument("--interval", type=float, default=60.0)

    args = parser.parse_args(argv)

    if args.cmd == "heartbeat":
        # Ensure directory for heartbeat file exists when recording
        if args.file:
            os.makedirs(os.path.dirname(args.file), exist_ok=True)
        record_heartbeat()
    elif args.cmd == "monitor":
        monitor_heartbeat(
            args.file,
            pid=args.pid,
            timeout=args.timeout,
            check_interval=args.interval,
        )
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()

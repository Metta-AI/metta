import re

from IPython.display import IFrame, display

from experiments.notebooks.utils.metrics import get_run


def show_replay(
    run_name: str, step: str | int = "last", width: int = 1000, height: int = 600
) -> None:
    run = get_run(run_name)
    if run is None:
        return

    replay_urls = fetch_replay_urls_for_run(run)

    if not replay_urls:
        print(f"No replays found for {run_name}")
        return

    # Select the requested replay
    if step == "last":
        selected = replay_urls[-1]
    elif step == "first":
        selected = replay_urls[0]
    else:
        # Find replay closest to requested step
        target_step = int(step)
        selected = min(replay_urls, key=lambda r: abs(r["step"] - target_step))
        if selected["step"] != target_step:
            print(
                f"Note: Requested step {target_step}, showing closest available step {selected['step']}"
            )

    print(f"Loading MettaScope viewer for {run_name} at step {selected['step']:,}...")
    print(f"\nDirect link: {selected['url']}")
    display(IFrame(src=selected["url"], width=width, height=height))


def get_available_replays(run_name: str) -> list[dict]:
    run = get_run(run_name)
    if run is None:
        return []

    return fetch_replay_urls_for_run(run)


def fetch_replay_urls_for_run(run) -> list[dict]:
    files = run.files()
    replay_urls = []

    # Filter for replay HTML files
    replay_files = [
        f
        for f in files
        if "media/html/replays/link_" in f.name and f.name.endswith(".html")
    ]

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
                        {
                            "step": step,
                            "url": href,
                            "filename": file.name,
                            "label": f"Step {step:,}",
                        }
                    )
        except Exception:
            pass

    return replay_urls

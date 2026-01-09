import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from metta.common.util.log_config import init_logging, suppress_noisy_logs
from mettagrid.policy.prepare_policy_spec import download_policy_spec_from_s3_as_zip
from mettagrid.util.uri_resolvers.schemes import resolve_uri

logger = logging.getLogger(__name__)

app = typer.Typer()


def atomic_copy(src: Path, dst: Path):
    fd, tmp_path = tempfile.mkstemp(dir=dst.parent)
    try:
        os.close(fd)
        shutil.copy(src, tmp_path)
        os.rename(tmp_path, dst)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@app.command()
def main(
    policy: Annotated[str, typer.Option(help="Policy ID")],
    output: Annotated[str, typer.Option(help="Output path")],
):
    """Download a policy zip file from the metta registry by ID."""
    uri = f"metta://policy/{policy}"

    logger.info("Resolving %s", uri)
    resolved = resolve_uri(uri)

    logger.info("Downloading from %s", resolved.canonical)
    cached_path = download_policy_spec_from_s3_as_zip(
        s3_path=resolved.canonical,
        remove_downloaded_copy_on_exit=True,
    )

    output_path = Path(output)
    if output_path.exists() and not output_path.is_file():
        logger.error("Cannot overwrite (%r exists and is not a regular file)", output_path)
        raise typer.Exit(code=1)

    atomic_copy(cached_path, output_path)

    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    app()

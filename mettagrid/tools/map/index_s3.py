import argparse
import logging
import os
import signal  # Aggressively exit on ctrl+c

from mettagrid.map.utils import s3utils, storage

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="S3 directory, e.g. s3://.../dir")
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    s3_dir = args.dir
    target = args.target or f"{s3_dir}/index.txt"

    uri_list = s3utils.list_objects(s3_dir)

    storage.save_to_uri(text="\n".join(uri_list), uri=target)
    logger.info(f"Index with {len(uri_list)} maps saved to {target}")


if __name__ == "__main__":
    main()

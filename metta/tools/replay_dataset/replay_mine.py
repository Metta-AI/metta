#!/usr/bin/env python3
"""
Mine replays from S3 to create supervised learning datasets.

This script:
1. Queries the stats database for replay URLs
2. Downloads and parses replay files from S3
3. Extracts (observation, action) pairs for supervised learning
4. Saves curated datasets back to S3
"""

import argparse
import json
import logging
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from metta.common.util.constants import (
    PROD_STATS_SERVER_URI,
    SOFTMAX_S3_REPLAYS_PREFIX,
)
from metta.common.util.file import local_copy

logger = logging.getLogger(__name__)


@dataclass
class ReplaySample:
    """A single training sample extracted from a replay."""

    observation: dict[str, Any]
    action: int
    agent_id: int
    episode_id: str
    timestep: int


class ReplayMiner:
    """Mines replays from S3 to create supervised learning datasets."""

    def __init__(
        self,
        stats_server_uri: str,
        date: str,  # YYYY-MM-DD format
        min_reward: float = 0.0,
        environment: str | None = None,
    ):
        self.stats_server_uri = stats_server_uri
        self.date = date
        self.min_reward = min_reward
        self.environment = environment
        self.samples: list[ReplaySample] = []

    def load_replay(self, replay_url: str) -> dict[str, Any] | None:
        """Load and decompress a replay file from S3 or local path.

        Returns None if replay is not version 2 (skipped).
        """
        with local_copy(replay_url) as local_path:
            compressed = local_path.read_bytes()
            decompressed = zlib.decompress(compressed)
            replay_data = json.loads(decompressed)

            # Validate version
            version = replay_data.get("version")
            if version != 2:
                logger.warning(f"Skipping replay {replay_url}: version {version} (only v2 supported)")
                return None

            return replay_data

    def extract_samples_from_replay(self, replay_data: dict[str, Any], episode_id: str) -> list[ReplaySample]:
        """Extract (observation, action) pairs from a single replay."""
        samples = []
        max_steps = replay_data["max_steps"]

        for obj in replay_data["objects"]:
            if not obj.get("is_agent"):
                continue

            agent_id = obj["agent_id"]

            # Extract time series data
            actions = self._extract_timeseries(obj.get("action_id", 0), max_steps)
            locations = self._extract_location_timeseries(obj.get("location", [0, 0, 0]), max_steps)

            # Create samples for each timestep
            for t in range(max_steps):
                obs = {
                    "location": locations[t],
                    "orientation": obj.get("orientation", 0),
                    "inventory": obj.get("inventory", []),
                    # Add more observation fields as needed
                }

                sample = ReplaySample(
                    observation=obs,
                    action=actions[t],
                    agent_id=agent_id,
                    episode_id=episode_id,
                    timestep=t,
                )
                samples.append(sample)

        return samples

    def _extract_timeseries(self, data: Any, max_steps: int) -> list[Any]:
        """Convert replay time series format to per-timestep list."""
        if not isinstance(data, list):
            # Static value - replicate for all timesteps
            return [data] * max_steps

        if len(data) == 0:
            return [0] * max_steps

        # Check if it's a time series [[step, value], ...]
        if isinstance(data[0], list) and len(data[0]) == 2:
            result = []
            current_value = data[0][1]
            step_idx = 0

            for t in range(max_steps):
                # Check if we need to update value
                if step_idx < len(data) - 1 and t >= data[step_idx + 1][0]:
                    step_idx += 1
                    current_value = data[step_idx][1]
                result.append(current_value)
            return result

        # Single value
        return [data] * max_steps

    def _extract_location_timeseries(self, data: Any, max_steps: int) -> list[list[float]]:
        """Extract location time series."""
        if isinstance(data, list) and len(data) == 3:
            # Static location
            return [data] * max_steps

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Time series format
            result = []
            current_loc = data[0][1]
            step_idx = 0

            for t in range(max_steps):
                if step_idx < len(data) - 1 and t >= data[step_idx + 1][0]:
                    step_idx += 1
                    current_loc = data[step_idx][1]
                result.append(current_loc)
            return result

        return [[0, 0, 0]] * max_steps

    def _get_replay_urls(self) -> list[str]:
        """Get replay URLs from either HTTP API or direct database access."""
        # Check if this is an HTTP(S) API endpoint
        if self.stats_server_uri.startswith("http://") or self.stats_server_uri.startswith("https://"):
            return self._get_replay_urls_from_api()
        return []

    def _get_replay_urls_from_api(self) -> list[str]:
        """Query replay URLs via HTTP stats API for a specific date."""
        from metta.app_backend.clients.stats_client import HttpStatsClient

        with HttpStatsClient(backend_url=self.stats_server_uri) as client:
            # Build SQL query to get replay URLs for the specified date
            query = f"""
                SELECT replay_url
                FROM episodes
                WHERE replay_url IS NOT NULL
                AND DATE(created_at) = '{self.date}'
            """

            if self.environment:
                query += f" AND env_name = '{self.environment}'"

            query += " ORDER BY created_at ASC"

            result = client.sql_query(query)

            # Extract URLs from query results
            replay_urls = [row[0] for row in result.rows if row[0]]
            return replay_urls

    def get_earliest_replay_date(self) -> str | None:
        """Query database for earliest replay date with replays."""
        if self.stats_server_uri.startswith("http://") or self.stats_server_uri.startswith("https://"):
            from metta.app_backend.clients.stats_client import HttpStatsClient

            with HttpStatsClient(backend_url=self.stats_server_uri) as client:
                query = """
                    SELECT MIN(DATE(created_at)) as earliest_date
                    FROM episodes
                    WHERE replay_url IS NOT NULL
                """
                result = client.sql_query(query)
                return result.rows[0][0] if result.rows and result.rows[0][0] else None

    def mine_replays(self) -> dict[str, Any]:
        """Query database, download replays, and extract samples for a specific date."""
        logger.info(f"Mining replays from {self.stats_server_uri}")
        logger.info(f"Date: {self.date}")
        logger.info(f"Filters: min_reward={self.min_reward}, environment={self.environment}")

        # Support both HTTP API and direct database access
        replay_urls = self._get_replay_urls()
        logger.info(f"Found {len(replay_urls)} replay URLs for {self.date}")

        processed_count = 0
        skipped_count = 0

        for i, replay_url in enumerate(replay_urls):
            try:
                logger.info(f"Processing replay {i + 1}/{len(replay_urls)}: {replay_url}")
                replay_data = self.load_replay(replay_url)

                if replay_data is None:
                    skipped_count += 1
                    continue

                episode_samples = self.extract_samples_from_replay(replay_data, f"episode_{i}")
                self.samples.extend(episode_samples)
                processed_count += 1
                logger.info(f"Extracted {len(episode_samples)} samples from replay {i + 1}")
            except Exception as e:
                logger.error(f"Error processing replay {replay_url}: {e}")
                continue

        logger.info(f"Processed {processed_count} replays, skipped {skipped_count} (version mismatch)")

        logger.info(f"Mining complete: {len(self.samples)} samples from {processed_count} episodes")

        # Return stats dict
        return {
            "num_episodes": processed_count,
            "num_samples": len(self.samples),
            "date": self.date,
        }

    def save_dataset(self, output_uri: str) -> None:
        """Save dataset to Parquet format (local or S3)."""
        import json

        # Create local directory if needed (S3 paths are handled by pandas automatically)
        if not output_uri.startswith("s3://"):
            output_path = Path(output_uri)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving to local path: {output_uri}")
        else:
            logger.info(f"Saving to S3: {output_uri}")

        # Handle empty sample set by creating DataFrame with expected schema
        if not self.samples:
            logger.warning(f"No samples to save for {self.date}, creating empty shard with schema")
            df = pd.DataFrame(
                columns=[
                    "observation",
                    "action",
                    "agent_id",
                    "episode_id",
                    "timestep",
                    "date",
                    "min_reward",
                    "environment",
                ]
            )
            # Set column types to match expected schema
            df = df.astype(
                {
                    "observation": "object",
                    "action": "int64",
                    "agent_id": "int64",
                    "episode_id": "object",
                    "timestep": "int64",
                    "date": "object",
                    "min_reward": "float64",
                    "environment": "object",
                }
            )
        else:
            # Convert samples to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "observation": json.dumps(sample.observation),  # Serialize dict to JSON string
                        "action": sample.action,
                        "agent_id": sample.agent_id,
                        "episode_id": sample.episode_id,
                        "timestep": sample.timestep,
                        "date": self.date,
                        "min_reward": self.min_reward,
                        "environment": self.environment,
                    }
                    for sample in self.samples
                ]
            )

        # Save to parquet (handles S3 automatically if URI is s3://)
        df.to_parquet(output_uri, index=False)
        logger.info(f"Saved {len(df)} samples to {output_uri}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Mine replays to create daily supervised learning datasets",
        epilog="Creates daily-sharded datasets from game replays stored in S3.",
    )
    parser.add_argument(
        "--stats-server-uri",
        type=str,
        default=PROD_STATS_SERVER_URI,
        help="Stats server api endpoint",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Single date to process in YYYY-MM-DD format (default: yesterday)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for range processing (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for range processing (YYYY-MM-DD, inclusive, default: yesterday)",
    )
    parser.add_argument(
        "--backfill-all",
        action="store_true",
        help="Process all dates from earliest replay to yesterday",
    )
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum episode reward to include")
    parser.add_argument("--environment", type=str, help="Filter by environment name (optional)")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=SOFTMAX_S3_REPLAYS_PREFIX,
        help="Output directory or S3 prefix (default: production S3 bucket)",
    )

    args = parser.parse_args()

    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Determine processing mode and date range
        if args.backfill_all:
            # Query for earliest date
            logger.info("Backfill mode: querying for earliest replay date...")
            temp_miner = ReplayMiner(
                stats_server_uri=args.stats_server_uri,
                date=yesterday,  # Temporary, just for query
                min_reward=args.min_reward,
                environment=args.environment,
            )
            start_date = temp_miner.get_earliest_replay_date()
            if not start_date:
                logger.error("No replays found in database")
                return 1
            end_date = args.end_date if args.end_date else yesterday
            logger.info(f"Backfilling from {start_date} to {end_date}")
        elif args.start_date:
            # Explicit range
            start_date = args.start_date
            end_date = args.end_date if args.end_date else yesterday
            logger.info(f"Processing date range: {start_date} to {end_date}")
        elif args.date:
            # Single date
            start_date = end_date = args.date
            logger.info(f"Processing single date: {start_date}")
        else:
            # Default: yesterday
            start_date = end_date = yesterday
            logger.info(f"No date specified, using yesterday: {yesterday}")

        # Generate list of dates to process
        dates_to_process = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        while current <= end:
            dates_to_process.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        logger.info(f"Total dates to process: {len(dates_to_process)}")

        # Process dates sequentially
        if len(dates_to_process) == 1:
            # Single date - process directly
            success = process_single_date(
                dates_to_process[0],
                args.stats_server_uri,
                args.min_reward,
                args.environment,
                args.output_prefix,
            )
            return 0 if success else 1
        else:
            # Multiple dates - sequential processing
            return process_date_range(
                dates_to_process,
                args.stats_server_uri,
                args.min_reward,
                args.environment,
                args.output_prefix,
            )

    except Exception as e:
        logger.error(f"Replay mining failed: {e}", exc_info=True)
        return 1


def process_single_date(
    date: str,
    stats_server_uri: str,
    min_reward: float,
    environment: str | None,
    output_prefix: str,
) -> bool:
    """Process a single date and return success status."""
    try:
        miner = ReplayMiner(
            stats_server_uri=stats_server_uri,
            date=date,
            min_reward=min_reward,
            environment=environment,
        )

        stats = miner.mine_replays()

        # Create dataset name from date (YYYY-MM-DD -> YYYYMMDD)
        date_str = date.replace("-", "")
        output_name = f"replays_{date_str}.parquet"
        output_uri = f"{output_prefix}/{output_name}"

        miner.save_dataset(output_uri)

        logger.info(f"✓ {date}: {stats['num_samples']} samples from {stats['num_episodes']} episodes")
        return True

    except Exception as e:
        logger.error(f"✗ {date}: Failed - {e}")
        return False


def process_date_range(
    dates: list[str],
    stats_server_uri: str,
    min_reward: float,
    environment: str | None,
    output_prefix: str,
) -> int:
    """Process multiple dates sequentially and return exit code."""
    logger.info(f"Processing {len(dates)} dates sequentially")

    successful = 0
    failed = 0

    # Process dates one by one
    for i, date in enumerate(dates, 1):
        logger.info(f"[{i}/{len(dates)}] Processing {date}...")
        try:
            success = process_single_date(
                date,
                stats_server_uri,
                min_reward,
                environment,
                output_prefix,
            )
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {date}: Unexpected error - {e}")
            failed += 1

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"Total dates: {len(dates)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

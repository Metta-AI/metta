"""PyTorch Dataset for loading replay datasets from Parquet files via DuckDB.

Usage:
    from metta.tools.replay_dataset import ReplayDataset, merge_datasets
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import duckdb
from torch.utils.data import Dataset

from metta.common.util.constants import SOFTMAX_S3_REPLAYS_PREFIX


class ReplayDataset(Dataset):
    """PyTorch Dataset that loads replay data from Parquet files using DuckDB.

    Supports:
    - Loading from S3 or local paths
    - Date range queries
    - SQL-based filtering
    - Automatic handling of multiple daily files
    """

    def __init__(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        dates: list[str] | None = None,
        base_path: str = SOFTMAX_S3_REPLAYS_PREFIX,
        filters: dict[str, str] | None = None,
    ):
        """Load replay datasets from Parquet files.

        Args:
            start_date: Start date (YYYY-MM-DD), inclusive
            end_date: End date (YYYY-MM-DD), inclusive
            dates: Specific list of dates to load (alternative to start/end)
            base_path: Base path for datasets (local or S3)
            filters: SQL filter conditions, e.g. {"agent_id": "IN (1, 2)", "action": "> 0"}

        Examples:
            # Load last week from S3
            dataset = ReplayDataset(start_date="2025-10-10", end_date="2025-10-17")

            # Load with filters
            dataset = ReplayDataset(
                start_date="2025-10-10",
                end_date="2025-10-17",
                filters={"agent_id": "= 5"}
            )

            # Load from local path
            dataset = ReplayDataset(
                start_date="2025-10-15",
                end_date="2025-10-17",
                base_path="./local_datasets"
            )
        """
        self.con = duckdb.connect()

        # Configure S3 access if needed
        if base_path.startswith("s3://"):
            self.con.execute("INSTALL httpfs")
            self.con.execute("LOAD httpfs")
            self.con.execute("SET s3_region='us-east-1'")

        # Build file pattern
        if dates:
            date_list = dates
        elif start_date and end_date:
            date_list = self._generate_date_range(start_date, end_date)
        else:
            raise ValueError("Must provide either (start_date, end_date) or dates")

        # Build query
        patterns = [f"{base_path}/replays_{d.replace('-', '')}.parquet" for d in date_list]
        pattern_clause = " UNION ALL ".join([f"SELECT * FROM '{p}'" for p in patterns])

        query = f"SELECT * FROM ({pattern_clause})"

        # Add filters
        if filters:
            where_clauses = [f"{k} {v}" for k, v in filters.items()]
            query += " WHERE " + " AND ".join(where_clauses)

        # Execute query and load data
        try:
            self.df = self.con.execute(query).df()
            print(f"Loaded {len(self.df)} samples from {len(date_list)} day(s)")
        except Exception as e:
            print(f"Error loading datasets: {e}")
            print(f"Attempted patterns: {patterns}")
            # Return empty dataframe
            import pandas as pd

            self.df = pd.DataFrame()

    def _generate_date_range(self, start: str, end: str) -> list[str]:
        """Generate list of dates between start and end (inclusive)."""
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        dates = []
        current = start_dt
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return dates

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample."""
        import json

        row = self.df.iloc[idx]
        return {
            "observation": json.loads(row["observation"]),  # Deserialize JSON string to dict
            "action": row["action"],
            "agent_id": row["agent_id"],
            "episode_id": row["episode_id"],
            "timestep": row["timestep"],
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Get dataset metadata from the data itself."""
        if len(self.df) == 0:
            return {}

        return {
            "num_samples": len(self.df),
            "dates": sorted(self.df["date"].unique().tolist()),
            "num_episodes": self.df["episode_id"].nunique(),
            "agents": self.df["agent_id"].unique().tolist(),
        }


def merge_datasets(
    start_date: str,
    end_date: str,
    output_path: str,
    base_path: str = SOFTMAX_S3_REPLAYS_PREFIX,
) -> None:
    """Merge multiple daily datasets into a single Parquet file.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Where to save merged dataset
        base_path: Where to find daily datasets

    Example:
        # Merge last week into single file
        merge_datasets(
            start_date="2025-10-10",
            end_date="2025-10-17",
            output_path="s3://softmax-public/datasets/replays_week42.parquet"
        )
    """
    dataset = ReplayDataset(start_date=start_date, end_date=end_date, base_path=base_path)

    # Save merged dataframe
    dataset.df.to_parquet(output_path, index=False)

    print(f"Saved merged dataset to {output_path}")
    print(f"Total samples: {len(dataset.df)}")
    print(f"Dates: {dataset.metadata['dates']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load or merge replay datasets from Parquet")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--merge", action="store_true", help="Merge datasets into single file")
    parser.add_argument("--output", type=str, help="Output path for merged dataset")
    parser.add_argument("--base-path", type=str, default=SOFTMAX_S3_REPLAYS_PREFIX, help="Base path for datasets")

    args = parser.parse_args()

    if args.merge:
        if not all([args.start_date, args.end_date, args.output]):
            parser.error("--merge requires --start-date, --end-date, and --output")

        merge_datasets(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output,
            base_path=args.base_path,
        )
    else:
        # Just load and show stats
        dataset = ReplayDataset(
            start_date=args.start_date,
            end_date=args.end_date,
            base_path=args.base_path,
        )

        print("\nDataset Statistics:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Metadata: {dataset.metadata}")

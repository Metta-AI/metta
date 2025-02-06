import duckdb
import os
import wandb

class WanDuckDb:
    def __init__(self, artifact_name: str, local_cache_dir: str = "./eval_cache"):
        self.artifact_name = artifact_name
        self.local_cache_dir = local_cache_dir
        os.makedirs(self.local_cache_dir, exist_ok=True)
        # Create a DuckDB in-memory connection
        self.conn = duckdb.connect(database=':memory:')

    def _download_artifacts(self):
        """Download all artifact versions for self.artifact_name from wandb."""
        api = wandb.Api()
        # List artifacts with the given name; adjust project and entity as needed.
        # For example, here we assume the current project and entity:
        artifacts = api.artifacts(type="eval_db", name=self.artifact_name)
        file_paths = []
        for art in artifacts:
            # Each artifact may have a version, download it
            local_path = art.download(root=self.local_cache_dir)
            # Assume that the JSON file is named "eval_stats.json" in each artifact
            json_file = os.path.join(local_path, "eval_stats.json")
            if os.path.exists(json_file):
                file_paths.append(json_file)
        return file_paths

    def query(self, sql: str):
        """Download artifacts if necessary, load them into DuckDB, and run the SQL query."""
        json_files = self._download_artifacts()
        if not json_files:
            raise ValueError("No evaluation artifacts found.")

        # DuckDB can query JSON files if we use the 'read_json_auto' function.
        # Here we union all the JSON data.
        union_query = " UNION ALL ".join(
            [f"SELECT * FROM read_json_auto('{fp}')" for fp in json_files]
        )
        full_query = f"WITH eval_data AS ({union_query}) {sql}"
        return self.conn.execute(full_query).fetchdf()

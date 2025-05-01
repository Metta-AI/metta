import logging
import os

import hydra
from omegaconf import DictConfig

from metta.sim.eval_stats_db import EvalStatsDB
from metta.util.config import pretty_print_config
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


@hydra.main(version_base=None, config_path="../configs", config_name="analyze_job")
def main(cfg: DictConfig) -> None:
    """
    Provides a SQL interface to query the evaluation stats database.
    """
    setup_mettagrid_environment(cfg)
    logger = logging.getLogger(__name__)
    
    # Print the configuration
    pretty_print_config(cfg)
    
    logger.info(f"Loading database from {cfg.eval_db_uri}")
    
    # Initialize WandbContext and load the database
    with WandbContext(cfg) as wandb_run:
        eval_stats_db = EvalStatsDB.from_uri(cfg.eval_db_uri, cfg.run_dir, wandb_run)
    
    # Get and display available metrics
    available_metrics = eval_stats_db.available_metrics
    logger.info(f"Available metrics: {len(available_metrics)}")
    if len(available_metrics) > 0:
        logger.info(f"Examples: {available_metrics[:5]}")
    
    # Access the DuckDB connection for interactive queries
    db_conn = eval_stats_db._db
    
    # Start DuckDB's interactive CLI
    logger.info("Starting DuckDB CLI. Type '.help' for help, '.quit' to exit.")
    logger.info("You can query the 'eval_data' table. For example: SELECT * FROM eval_data LIMIT 10;")
    
    try:
        # Execute DuckDB's interactive shell
        db_conn.execute("CALL sql_mode()")
    except Exception as e:
        logger.error(f"Error starting SQL mode: {e}")
        
        # Fallback to manual query input if sql_mode() isn't available
        logger.info("Falling back to manual query input. Type 'exit' to quit.")
        
        while True:
            query = input("\nSQL> ")
            if query.lower() == 'exit':
                break
                
            try:
                if query.lower().startswith('show columns'):
                    print("\n".join(available_metrics))
                    continue
                    
                result = db_conn.execute(query).fetchdf()
                print(result.to_string())
                
                if len(result) > 20:
                    logger.info(f"... {len(result) - 20} more rows")
                    
            except Exception as e:
                logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()

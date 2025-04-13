"""
Generate reports from policy evaluation metrics.
"""

import logging
import os
import shutil
import tempfile

import boto3
from botocore.exceptions import NoCredentialsError
from omegaconf import DictConfig

from eval.db import PolicyEvalDB
from eval.heatmap import create_matrix_visualization

logger = logging.getLogger(__name__)

def upload_to_s3(content: str, s3_path: str) -> bool:
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")
    
    s3_parts = s3_path[5:].split("/", 1)
    if len(s3_parts) < 2:
        raise ValueError(f"Invalid S3 path: {s3_path}. Must be in format s3://bucket/path")
    
    bucket = s3_parts[0]
    key = s3_parts[1]
    
    try:
        s3_client = boto3.client('s3')
        logger.info(f"Uploading content to S3 bucket {bucket}, key {key}")
        s3_client.put_object(
            Body=content,
            Bucket=bucket,
            Key=key,
            ContentType='text/html'
        )
        logger.info(f"Successfully uploaded to {s3_path}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Make sure AWS credentials are configured.")
        return False
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return False

def generate_report_html(
    db: PolicyEvalDB,
    metric: str = "episode_reward",
    title: str = "Policy Evaluation Report"
) -> str:
    logger.info(f"Generating matrix visualization for metric: {metric}")
    matrix_data = db.get_matrix_data(metric)
    
    if matrix_data.empty:
        logger.warning(f"No data found for metric: {metric}")
        return "<html><body><h1>No data available</h1></body></html>"
    
    # Create visualization
    fig = create_matrix_visualization(
        matrix_data=matrix_data,
        metric=metric,
        colorscale='RdYlGn',
        height=600,
        width=900
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div id="heatmap"></div>
        </div>
        <script>
            var figure = {fig.to_json()};
            Plotly.newPlot('heatmap', figure.data, figure.layout);
        </script>
    </body>
    </html>
    """
    
    return html_content

def generate_report(
    cfg: DictConfig,
):
    db_path = cfg.analyzer.analysis.get("db_path")
    output_path = cfg.analyzer.analysis.get("output_path")
    metric = cfg.analyzer.analysis.metrics[0].metric
    
    if metric is None:
        raise ValueError("Metric is not specified")
    if output_path is None:
        raise ValueError("Output path is not specified")

    # Use temporary directory for database if db_path not specified
    tmp_dir = None
    if db_path is None:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "policy_metrics.sqlite")
        logger.info(f"Using temporary database path: {db_path}")
    
    try:
        # Initialize database and import data
        logger.info(f"Initializing database at {db_path}")
        db = PolicyEvalDB(db_path)
        db.import_from_eval_stats(cfg)
        
        # Generate the HTML report
        html_content = generate_report_html(
            db,
            metric=metric,
            title=f"Policy Evaluation Report: {metric}"
        )
        
        # Use output_path from config if not provided
        if output_path is None and hasattr(cfg.analyzer, "analysis"):
            output_path = cfg.analyzer.analysis.get("output_path")
        
        # Handle output based on output_path
        if output_path:
            if output_path.startswith("s3://"):
                # Upload directly to S3
                upload_to_s3(html_content, output_path)
                logger.info(f"Report uploaded to {output_path}")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(html_content)
                logger.info(f"Report saved to {output_path}")
            return html_content, output_path
        else:
            # Just return the HTML
            logger.info("No output path specified. HTML report generated but not saved.")
            return html_content
            
    finally:
        # Clean up temporary directory if created
        if tmp_dir and os.path.exists(tmp_dir):
            logger.info(f"Cleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)
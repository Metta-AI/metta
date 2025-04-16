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

from metta.eval.db import PolicyEvalDB
from metta.eval.heatmap import create_matrix_visualization

logger = logging.getLogger(__name__)


def upload_to_s3(content: str, s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")

    s3_parts = s3_path[5:].split("/", 1)
    if len(s3_parts) < 2:
        raise ValueError(f"Invalid S3 path: {s3_path}. Must be in format s3://bucket/path")

    bucket = s3_parts[0]
    key = s3_parts[1]

    try:
        s3_client = boto3.client("s3")
        logger.info(f"Uploading content to S3 bucket {bucket}, key {key}")
        s3_client.put_object(Body=content, Bucket=bucket, Key=key, ContentType="text/html")
        logger.info(f"Successfully uploaded to {s3_path}")
    except NoCredentialsError as e:
        logger.error("AWS credentials not found. Make sure AWS credentials are configured. Try running setup_sso.py.")
        raise e
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        raise e


def generate_report_html(cfg: DictConfig) -> str:
    metric = cfg.analyzer.metric
    view_type = cfg.analyzer.view_type
    policy_uri = cfg.analyzer.policy_uri

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "policy_metrics.sqlite")
    logger.info(f"Using temporary database path: {db_path}")

    try:
        # Initialize database and import data
        logger.info(f"Initializing database at {db_path}")
        db = PolicyEvalDB(db_path)
        db.import_from_eval_stats(cfg)

        # Generate report title with additional context for policy-specific views
        title = f"Policy Evaluation Report: {metric}"

        logger.info(f"Generating matrix visualization for metric: {metric} with view type: {view_type}")

        matrix_data = db.get_matrix_data(metric, view_type=view_type, policy_uri=policy_uri)

        if matrix_data.empty:
            logger.warning(f"No data found for metric: {metric}")
            return "<html><body><h1>No data available</h1></body></html>"

        score_range = (0, 1)
        RED = "rgb(235, 40, 40)"
        YELLOW = "rgb(225, 210, 80)"
        LIGHT_GREEN = "rgb(175, 230, 80)"
        FULL_GREEN = "rgb(20, 230, 80)"
        colorscale = [[0.0, RED], [0.5 / 3.0, RED], [2.2 / 3.0, YELLOW], [2.8 / 3.0, LIGHT_GREEN], [1.0, FULL_GREEN]]

        # Create visualization with fixed score range and custom colorscale
        fig = create_matrix_visualization(
            matrix_data=matrix_data,
            metric=metric,
            colorscale=colorscale,
            score_range=score_range,
            height=600,
            width=900,
        )

        view_type_description = ""
        if view_type == "policy_versions" and policy_uri:
            view_type_description = f" - All versions of {policy_uri}"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}{view_type_description}</title>
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
                <h1>{title}{view_type_description}</h1>
                <div id="heatmap"></div>
            </div>
            <script>
                var figure = {fig.to_json()};
                Plotly.newPlot('heatmap', figure.data, figure.layout);
            </script>
        </body>
        </html>
        """
    finally:
        # Clean up temporary directory if created
        if os.path.exists(tmp_dir):
            logger.info(f"Cleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)

    return html_content


def generate_report(cfg: DictConfig):
    output_path = cfg.analyzer.output_path
    view_type = cfg.analyzer.view_type
    policy_uri = cfg.analyzer.policy_uri
    # Generate the HTML report
    html_content = generate_report_html(cfg)

    # Add policy name to output path if we're doing a policy-specific report
    if view_type == "policy_versions" and policy_uri:
        # Extract policy name from URI if needed
        policy_filename = policy_uri.split("/")[-1].replace(":", "_")
        filename, ext = os.path.splitext(output_path)
        output_path = f"{filename}_{policy_filename}{ext}"

    if output_path.startswith("s3://"):
        # Upload directly to S3
        upload_to_s3(html_content, output_path)
        logger.info(f"Report uploaded to {output_path}")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        logger.info(f"Report saved to {output_path}")
    return html_content, output_path

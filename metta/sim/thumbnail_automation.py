"""
Thumbnail generation automation for MettaGrid simulations.

This module handles the automation logic for generating and uploading thumbnails
during simulation runs. It coordinates between the core thumbnail generation
library and the project's file I/O utilities.
"""

import logging
import os

from metta.map.utils.thumbnail import generate_thumbnail_from_replay
from metta.mettagrid.util import file as file_utils

logger = logging.getLogger(__name__)

# Configuration from environment variables
THUMBNAIL_GENERATION_ENABLED = os.getenv("METTA_THUMBNAIL_GENERATION_ENABLED", "true").lower() == "true"
THUMBNAIL_FORCE_REGENERATE = os.getenv("METTA_THUMBNAIL_FORCE_REGENERATE", "false").lower() == "true"

# S3 configuration - matches existing thumbnail locations
THUMBNAIL_S3_BUCKET = "softmax-public"
THUMBNAIL_S3_PREFIX = "policydash/evals/img"


def eval_name_to_s3_key(eval_name: str) -> str:
    """
    Convert eval_name to S3 key using naming scheme that matches frontend expectations.

    Args:
        eval_name: Evaluation name like "navigation/emptyspace_withinsight"

    Returns:
        S3 key like "navigation_emptyspace_withinsight.png"
    """
    return f"{eval_name.replace('/', '_').lower()}.png"


def eval_name_to_s3_uri(eval_name: str) -> str:
    """
    Convert eval_name to full S3 URI for use with mettagrid.util.file functions.

    Args:
        eval_name: Evaluation name like "navigation/emptyspace_withinsight"

    Returns:
        Full S3 URI like "s3://softmax-public/policydash/evals/img/navigation_emptyspace_withinsight.png"
    """
    s3_key = eval_name_to_s3_key(eval_name)
    return f"s3://{THUMBNAIL_S3_BUCKET}/{THUMBNAIL_S3_PREFIX}/{s3_key}"


def should_generate_thumbnail(eval_name: str) -> bool:
    """
    Check if thumbnail should be generated for this eval_name.

    Args:
        eval_name: Evaluation name to check

    Returns:
        True if thumbnail should be generated, False if it already exists or generation is disabled
    """
    if not THUMBNAIL_GENERATION_ENABLED:
        logger.debug(f"Thumbnail generation disabled for {eval_name}")
        return False

    if THUMBNAIL_FORCE_REGENERATE:
        logger.debug(f"Force regeneration enabled for {eval_name}")
        return True

    # Check if thumbnail already exists using project's file utilities
    s3_uri = eval_name_to_s3_uri(eval_name)

    try:
        if file_utils.exists(s3_uri):
            logger.debug(f"Thumbnail already exists for {eval_name}: {s3_uri}")
            return False
        else:
            logger.debug(f"Thumbnail does not exist for {eval_name}, should generate")
            return True

    except Exception as e:
        logger.warning(f"Error checking thumbnail existence for {eval_name}: {e}")
        # On error, assume it exists to avoid duplicate generation
        return False


def upload_thumbnail_to_s3(thumbnail_data: bytes, eval_name: str) -> bool:
    """
    Upload thumbnail data to S3 using project's file utilities.

    Args:
        thumbnail_data: PNG image data as bytes
        eval_name: Evaluation name for S3 key generation

    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        s3_uri = eval_name_to_s3_uri(eval_name)

        # Use project's standard file utilities for S3 upload
        file_utils.write_data(s3_uri, thumbnail_data, content_type="image/png")

        logger.info(f"Uploaded thumbnail for {eval_name} to {s3_uri}")
        return True

    except Exception as e:
        logger.error(f"Failed to upload thumbnail for {eval_name}: {e}")
        return False


def maybe_generate_and_upload_thumbnail(replay_data: dict, eval_name: str) -> bool:
    """
    Main automation entry point: generate and upload thumbnail if needed.

    This function is called from simulation.py after a simulation completes.
    It checks if a thumbnail is needed, generates it using the core library,
    and uploads it using the project's file utilities.

    Args:
        replay_data: Replay data from simulation's episode writer
        eval_name: Evaluation name from simulation

    Returns:
        True if thumbnail was generated and uploaded successfully, False otherwise
    """
    if not should_generate_thumbnail(eval_name):
        return False

    try:
        # Generate thumbnail using core library
        logger.info(f"Generating thumbnail for {eval_name}")
        thumbnail_data = generate_thumbnail_from_replay(replay_data)

        # Upload using project file utilities
        success = upload_thumbnail_to_s3(thumbnail_data, eval_name)

        if success:
            logger.info(f"Successfully generated and uploaded thumbnail for {eval_name}")
        else:
            logger.error(f"Thumbnail generation succeeded but upload failed for {eval_name}")

        return success

    except Exception as e:
        logger.error(f"Thumbnail generation failed for {eval_name}: {e}")
        return False

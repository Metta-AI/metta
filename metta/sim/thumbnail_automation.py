import logging
import os

from softmax.lib.utils import file as file_utils
from mettagrid.mapgen.utils.thumbnail import generate_thumbnail_from_replay

logger = logging.getLogger(__name__)

# Configuration from environment variables
THUMBNAIL_GENERATION_ENABLED = os.getenv("METTA_THUMBNAIL_GENERATION_ENABLED", "true").lower() == "true"

# S3 configuration - matches existing thumbnail locations
THUMBNAIL_S3_BUCKET = "softmax-public"
THUMBNAIL_S3_PREFIX = "policydash/evals/img"


def episode_id_to_s3_key(episode_id: str) -> str:
    """
    Convert episode_id to S3 key using unique episode-based naming (like replay files).

    Args:
        episode_id: Unique episode ID (UUID) like "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    Returns:
        S3 key like "a1b2c3d4-e5f6-7890-abcd-ef1234567890.png"
    """
    return f"{episode_id}.png"


def episode_id_to_s3_uri(episode_id: str) -> str:
    """
    Convert episode_id to full S3 URI for use with softmax.lib.utils helpers.

    Args:
        episode_id: Unique episode ID (UUID) like "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    Returns:
        Full S3 URI like "s3://softmax-public/policydash/evals/img/a1b2c3d4-e5f6-7890-abcd-ef1234567890.png"
    """
    s3_key = episode_id_to_s3_key(episode_id)
    return f"s3://{THUMBNAIL_S3_BUCKET}/{THUMBNAIL_S3_PREFIX}/{s3_key}"


def upload_thumbnail_to_s3(thumbnail_data: bytes, episode_id: str) -> tuple[bool, str | None]:
    """
    Upload thumbnail data to S3 using project's file utilities.

    Args:
        thumbnail_data: PNG image data as bytes
        episode_id: Unique episode ID for S3 key generation

    Returns:
        Tuple of (success: bool, http_thumbnail_url: str | None)
    """
    try:
        s3_uri = episode_id_to_s3_uri(episode_id)

        # Use project's standard file utilities for S3 upload
        file_utils.write_data(s3_uri, thumbnail_data, content_type="image/png")

        # Convert S3 URI to HTTP URL for database storage
        http_thumbnail_url = file_utils.http_url(s3_uri)

        return True, http_thumbnail_url

    except Exception as e:
        logger.error(f"Failed to upload thumbnail for episode {episode_id}: {e}")
        return False, None


def maybe_generate_and_upload_thumbnail(replay_data: dict, episode_id: str) -> tuple[bool, str | None]:
    """
    Main automation entry point: generate and upload thumbnail for episode.

    This function is called from simulation.py after a simulation completes.
    It generates a unique thumbnail for each episode (like replay files),
    eliminating the need for conflict checking or shared naming schemes.

    Args:
        replay_data: Replay data from simulation's episode writer
        episode_id: Unique episode ID from simulation

    Returns:
        Tuple of (success: bool, thumbnail_url: str | None)
    """
    if not THUMBNAIL_GENERATION_ENABLED:
        logger.debug(f"Thumbnail generation disabled for episode {episode_id}")
        return False, None

    try:
        # Generate thumbnail using core library
        logger.debug(f"Generating thumbnail for episode {episode_id}")
        thumbnail_data = generate_thumbnail_from_replay(replay_data)

        # Upload using project file utilities
        success, thumbnail_url = upload_thumbnail_to_s3(thumbnail_data, episode_id)

        if success:
            logger.debug(f"Successfully generated and uploaded thumbnail for episode {episode_id}")
        else:
            logger.error(f"Thumbnail generation succeeded but upload failed for episode {episode_id}")

        return success, thumbnail_url

    except Exception as e:
        logger.error(f"Thumbnail generation failed for episode {episode_id}: {e}")
        return False, None

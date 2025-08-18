#!/usr/bin/env python3
"""
Export filesets to Google Drive for Asana AI consumption.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from aggregate import discover_files, build_markdown
from gdrive_io import DriveManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MANIFEST_FILE = Path("tools/gdrive_upload/manifest.json")
FILESETS_CONFIG = Path("tools/gdrive_upload/filesets.yml")


def load_config() -> Dict[str, Any]:
    """Load filesets configuration from YAML, with optional local overrides."""
    if not FILESETS_CONFIG.exists():
        raise FileNotFoundError(f"Configuration file not found: {FILESETS_CONFIG}")

    # Load base configuration
    with open(FILESETS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    # Load local overrides if they exist
    local_config_path = FILESETS_CONFIG.parent / "filesets.local.yml"
    if local_config_path.exists():
        logger.info("Loading local configuration overrides")
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)

        # Merge configurations (local overrides base)
        if 'drive' in local_config:
            config['drive'].update(local_config['drive'])
        if 'filesets' in local_config:
            config['filesets'].update(local_config['filesets'])

    return config


def load_manifest() -> Dict[str, Any]:
    """Load manifest from JSON file."""
    if not MANIFEST_FILE.exists():
        return {}

    with open(MANIFEST_FILE, 'r') as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Save manifest to JSON file."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)


def compute_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def export_fileset(fileset_name: str, config: Dict[str, Any], manifest: Dict[str, Any],
                  drive_manager: Optional[DriveManager], dry_run: bool = False,
                  seen_files: Optional[set] = None) -> str:
    """
    Export a single fileset to Google Drive.

    Returns the web link to the uploaded document.
    """
    logger.info(f"Processing fileset: {fileset_name}")

    # Get fileset configuration
    fileset_config = config['filesets'].get(fileset_name)
    if not fileset_config:
        raise ValueError(f"Fileset '{fileset_name}' not found in configuration")

    includes = fileset_config.get('includes', [])
    excludes = fileset_config.get('excludes', [])

    # Discover files
    files = discover_files(includes, excludes)

    # Apply deduplication if tracking seen files
    original_count = len(files)
    skipped_files = []

    if seen_files is not None:
        # Filter out files we've already processed
        unique_files = []
        for file_path in files:
            resolved_path = file_path.resolve()
            if resolved_path in seen_files:
                skipped_files.append(file_path)
            else:
                unique_files.append(file_path)
                seen_files.add(resolved_path)
        files = unique_files

        if skipped_files:
            logger.info(f"Deduplicated: {len(skipped_files)} files already seen in previous filesets")

    logger.info(f"Found {len(files)} files ({original_count} before deduplication)")

    # Early return for dry run - no Drive operations needed
    if dry_run:
        total_size = sum(f.stat().st_size for f in files if f.exists())
        logger.info(f"Total size: {total_size:,} bytes")
        for file in files[:10]:  # Show first 10 files
            logger.info(f"  {file}")
        if len(files) > 10:
            logger.info(f"  ... and {len(files) - 10} more files")
        return ""

    # Ensure we have a drive manager for actual uploads
    if drive_manager is None:
        raise ValueError("DriveManager is required for non-dry-run operations")

    # Build aggregated markdown
    content_bytes = build_markdown(fileset_name, files)
    current_hash = compute_hash(content_bytes)

    # Check if content has changed
    fileset_entry = manifest.get(fileset_name, {})
    if fileset_entry.get('last_hash') == current_hash:
        logger.info(f"Content unchanged for {fileset_name}, using existing URL")
        return fileset_entry['web_link']

    # Upload to Google Drive
    title = f"Fileset: {fileset_name}"
    existing_file_id = fileset_entry.get('file_id')

    file_id, web_link = drive_manager.create_or_update_document(
        title=title,
        content_bytes=content_bytes,
        existing_file_id=existing_file_id
    )

    # Update manifest
    manifest[fileset_name] = {
        'file_id': file_id,
        'web_link': web_link,
        'last_hash': current_hash
    }
    save_manifest(manifest)

    logger.info(f"Successfully exported {fileset_name}")
    logger.info(f"URL: {web_link}")

    return web_link


def main():
    parser = argparse.ArgumentParser(description="Export filesets to Google Drive")
    parser.add_argument("--fileset", help="Export a specific fileset")
    parser.add_argument("--all", action="store_true", help="Export all filesets")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without uploading")
    parser.add_argument("--no-deduplicate", action="store_true",
                       help="When using --all, allow duplicate files across filesets (default is to deduplicate)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.fileset and not args.all:
        parser.error("Must specify either --fileset or --all")

    try:
        # Load configuration and manifest
        config = load_config()
        manifest = load_manifest()

        # Initialize Drive manager (unless dry run)
        drive_manager = None
        if not args.dry_run:
            drive_manager = DriveManager(config['drive'])

        # Export filesets
        seen_files = set() if args.all and not args.no_deduplicate else None

        if args.all:
            filesets = list(config['filesets'].keys())
            if args.no_deduplicate:
                logger.info(f"Exporting all filesets (allowing duplicates): {', '.join(filesets)}")
            else:
                logger.info(f"Exporting all filesets (with deduplication): {', '.join(filesets)}")
        else:
            filesets = [args.fileset]
            if args.no_deduplicate:
                logger.warning("--no-deduplicate flag ignored when exporting single fileset")

        for fileset_name in filesets:
            try:
                web_link = export_fileset(
                    fileset_name, config, manifest, drive_manager, args.dry_run, seen_files
                )
                if web_link:
                    print(f"{fileset_name}: {web_link}")
            except Exception as e:
                logger.error(f"Failed to export {fileset_name}: {e}")
                if args.all:
                    continue  # Continue with other filesets
                else:
                    sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

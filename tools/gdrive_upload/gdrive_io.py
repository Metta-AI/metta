"""
Google Drive integration for fileset exports.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = Path("tools/gdrive_upload/token.json")
CREDENTIALS_FILE = Path("tools/gdrive_upload/credentials.json")


class DriveManager:
    """Manages Google Drive operations for fileset exports."""

    def __init__(self, drive_config: Dict[str, Any]):
        self.drive_config = drive_config
        self.service = self._get_service()
        self.parent_folder_id = self._ensure_parent_folder()

    def _get_service(self):
        """Get authenticated Google Drive service."""
        creds = None

        # Load existing token
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

        # If no valid credentials, request authorization
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            else:
                if not CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {CREDENTIALS_FILE}\n"
                        "Please download OAuth 2.0 credentials from Google Cloud Console."
                    )

                logger.info("Starting OAuth flow")
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for next run
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())

        return build('drive', 'v3', credentials=creds)

    def _ensure_parent_folder(self) -> str:
        """Ensure parent folder exists and return its ID."""
        folder_name = self.drive_config.get('parent_folder_name', 'Metta Exports')

        # Check if folder ID is explicitly configured
        if 'parent_folder_id' in self.drive_config:
            return self.drive_config['parent_folder_id']

        # Search for existing folder
        try:
            results = self.service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)"
            ).execute()

            folders = results.get('files', [])
            if folders:
                folder_id = folders[0]['id']
                logger.info(f"Using existing folder: {folder_name} (ID: {folder_id})")
                return folder_id

        except HttpError as e:
            logger.warning(f"Error searching for folder: {e}")

        # Create new folder
        logger.info(f"Creating new folder: {folder_name}")
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }

        try:
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            folder_id = folder['id']
            logger.info(f"Created folder: {folder_name} (ID: {folder_id})")
            return folder_id

        except HttpError as e:
            logger.error(f"Failed to create folder: {e}")
            raise

    def create_or_update_document(self, title: str, content_bytes: bytes,
                                existing_file_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Create or update a Google Doc with the given content.

        Args:
            title: Document title
            content_bytes: Document content as bytes
            existing_file_id: ID of existing file to update (if any)

        Returns:
            Tuple of (file_id, web_link)
        """
        media = MediaInMemoryUpload(content_bytes, mimetype="text/plain", resumable=False)

        if existing_file_id:
            # Update existing document
            logger.info(f"Updating existing document: {title}")
            try:
                file = self.service.files().update(
                    fileId=existing_file_id,
                    media_body=media,
                    fields="id, webViewLink"
                ).execute()

                file_id = file['id']
                web_link = file.get('webViewLink')
                logger.info(f"Updated document: {web_link}")
                return file_id, web_link

            except HttpError as e:
                if e.resp.status == 404:
                    logger.warning(f"Existing file not found, creating new one: {existing_file_id}")
                    existing_file_id = None  # Fall through to create new
                else:
                    raise

        if not existing_file_id:
            # Create new document
            logger.info(f"Creating new document: {title}")
            metadata = {
                'name': title,
                'parents': [self.parent_folder_id],
                'mimeType': 'application/vnd.google-apps.document'
            }

            try:
                file = self.service.files().create(
                    body=metadata,
                    media_body=media,
                    fields="id, webViewLink"
                ).execute()

                file_id = file['id']
                web_link = file.get('webViewLink')

                # Set permissions to "anyone with link can view"
                self._ensure_anyone_viewer(file_id)

                logger.info(f"Created document: {web_link}")
                return file_id, web_link

            except HttpError as e:
                logger.error(f"Failed to create document: {e}")
                raise

    def _ensure_anyone_viewer(self, file_id: str) -> None:
        """Set file permissions to 'anyone with link can view'."""
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }

            self.service.permissions().create(
                fileId=file_id,
                body=permission,
                fields='id'
            ).execute()

            logger.debug(f"Set public read permissions for file: {file_id}")

        except HttpError as e:
            if e.resp.status == 403:
                logger.warning(
                    "Failed to set public permissions - may be restricted by domain policy. "
                    "Document will be private to your account."
                )
            else:
                logger.error(f"Error setting permissions: {e}")
                # Don't raise - the document was created successfully

    def _retry_on_rate_limit(self, func, *args, **kwargs):
        """Execute function with exponential backoff on rate limits."""
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                if e.resp.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(wait_time)
                else:
                    raise

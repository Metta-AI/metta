import sys
from pathlib import Path

import pytest

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metta.mettagrid.util.file import (
    GOOGLE_DRIVE_CREDENTIALS_FILE,
    GOOGLE_DRIVE_TOKEN_FILE,
    exists,
    http_url,
    read,
    write_data,
)


def _has_gdrive_credentials():
    """Check if Google Drive credentials are available"""

    cred_path = Path(GOOGLE_DRIVE_CREDENTIALS_FILE).expanduser()
    token_path = Path(GOOGLE_DRIVE_TOKEN_FILE).expanduser()
    return cred_path.exists() or token_path.exists()


@pytest.mark.skipif(not _has_gdrive_credentials(), reason="Google Drive credentials not available")
def test_gdrive_write_read_file():
    """Test basic write/read cycle for gdrive://file/ID pattern"""
    test_data = b"hello gdrive"
    file_id = "1b9wLccQT1dtVt6uvOfPFlSlyzw_4SSJs"  # This is Justin's test file ID. Replace with relevant ID if needed.

    write_data(f"gdrive://file/{file_id}", test_data)
    result = read(f"gdrive://file/{file_id}")
    assert result == test_data


@pytest.mark.skipif(not _has_gdrive_credentials(), reason="Google Drive credentials not available")
def test_gdrive_write_read_folder():
    """Test write/read for gdrive://folder/ID/filename pattern"""
    test_data = b"hello folder"
    folder_id = (
        "1PdW-fbbHlCnleFIdMD0S7Tohy0VLsuV3"  # This is Justin's mettatest ID. Replace with relevant ID if needed.
    )

    write_data(f"gdrive://folder/{folder_id}/test.txt", test_data)
    result = read(f"gdrive://folder/{folder_id}/test.txt")
    assert result == test_data


@pytest.mark.skipif(not _has_gdrive_credentials(), reason="Google Drive credentials not available")
def test_gdrive_exists():
    """Test exists() function for Google Drive files"""
    file_id = "1b9wLccQT1dtVt6uvOfPFlSlyzw_4SSJs"  # This is Justin's test file ID. Replace with relevant ID if needed.
    assert exists(f"gdrive://file/{file_id}")
    assert not exists("gdrive://file/nonexistent_id")


@pytest.mark.skipif(not _has_gdrive_credentials(), reason="Google Drive credentials not available")
def test_gdrive_http_url():
    """Test URL conversion works"""
    url = http_url("gdrive://file/1ABC123")
    assert "drive.google.com" in url
    assert "1ABC123" in url

# This is a test file for the arxiv_ingest module.
# It uses pytest to check that the ingestion logic works as expected.
# The comments here are detailed to help you understand and edit the tests.

import os
import sys
import tempfile  # For creating temporary directories for PDF download tests
import pytest    # The main testing framework
from pathlib import Path
import importlib.util  # For importing a module by file path

# --- Import the arxiv_ingest module directly from its file ---
# This is a dynamic import, which means we load the module by its path.
# This is useful if the module isn't installed as a package yet.
spec = importlib.util.spec_from_file_location(
    "arxiv_ingest",
    os.path.join(os.path.dirname(__file__), "../src/metta/app_backend/arxiv_ingest.py")
)
# Check that the spec and loader are valid (prevents runtime errors)
if spec is None or spec.loader is None:
    raise ImportError("Could not load arxiv_ingest module spec or loader.")
# Create a new module object and execute the code in it
arxiv_ingest = importlib.util.module_from_spec(spec)
sys.modules["arxiv_ingest"] = arxiv_ingest
spec.loader.exec_module(arxiv_ingest)

# Now we can access the functions and classes from arxiv_ingest
PaperMetadata = arxiv_ingest.PaperMetadata
fetch_arxiv_paper = arxiv_ingest.fetch_arxiv_paper
InvalidArxivIDError = arxiv_ingest.InvalidArxivIDError
ArxivFetchError = arxiv_ingest.ArxivFetchError

# --- Test: Happy path (integration) ---
# This test checks that fetching a real arXiv paper works and returns valid metadata.
# The @pytest.mark.integration decorator lets you run or skip integration tests easily.
@pytest.mark.integration
def test_fetch_arxiv_paper_happy_path():
    """
    Test fetching metadata for a real arXiv paper.
    This test will make a real network request to arXiv.
    """
    arxiv_id = "1804.02464"  # Example paper ID
    meta = fetch_arxiv_paper(arxiv_id)
    # Check that the result is a PaperMetadata object
    assert isinstance(meta, PaperMetadata)
    # Check that the fields are populated as expected
    assert meta.arxiv_id == arxiv_id
    assert meta.title
    assert meta.authors
    assert meta.abstract
    assert meta.primary_category
    assert meta.categories
    assert meta.published
    assert meta.pdf_url.startswith("https://arxiv.org/pdf/")
    assert meta.metadata_fetched_at
    # DOI is optional, so just check type if present
    assert meta.doi is None or isinstance(meta.doi, str)
    # PDF path should be None unless we download
    assert meta.pdf_path is None

# --- Test: Invalid arXiv ID raises error ---
def test_invalid_arxiv_id_raises():
    """
    Test that an invalid arXiv ID raises the correct error.
    This checks our input validation logic.
    """
    with pytest.raises(InvalidArxivIDError):
        fetch_arxiv_paper("not_a_valid_id")

# --- Test: PDF download sanity (integration) ---
# This test checks that the PDF is actually downloaded and is not empty.
@pytest.mark.integration
def test_pdf_download_sanity():
    """
    Test that downloading a PDF works and the file is not empty.
    Uses a temporary directory so it doesn't clutter your workspace.
    """
    arxiv_id = "1804.02464"
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = fetch_arxiv_paper(arxiv_id, download_pdf=True, dest_dir=Path(tmpdir))
        # The pdf_path should be set and the file should exist
        assert meta.pdf_path is not None
        assert meta.pdf_path.exists()
        # The file should be non-empty (real PDF)
        assert meta.pdf_path.stat().st_size > 0
        # Clean up the file (not strictly needed, as tempdir is deleted)
        meta.pdf_path.unlink() 
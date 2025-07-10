"""
arXiv paper ingestion module.

This module provides functionality to fetch paper metadata and optionally download PDFs
from arXiv using the official API.
"""

# Standard library imports
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

# Third-party imports
import feedparser
import requests
from pydantic import BaseModel, Field

# Set up logging so you can see what's happening as the script runs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a data model for the paper metadata using Pydantic
class PaperMetadata(BaseModel):
    """
    This class defines the structure of the metadata we want to extract from arXiv.
    Each attribute is a field in the metadata. Pydantic will validate types for us.
    """
    arxiv_id: str = Field(..., description="arXiv identifier (e.g., '1804.02464')")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    primary_category: str = Field(..., description="Primary arXiv category")
    categories: List[str] = Field(..., description="All arXiv categories")
    published: datetime = Field(..., description="Publication date")
    doi: Optional[str] = Field(None, description="DOI if available")
    pdf_url: str = Field(..., description="URL to the PDF")
    metadata_fetched_at: datetime = Field(default_factory=datetime.utcnow, description="When metadata was fetched")
    pdf_path: Optional[Path] = Field(None, description="Local path to downloaded PDF if available")

# Custom error for invalid arXiv IDs
class InvalidArxivIDError(ValueError):
    """
    Raised when an invalid arXiv ID is provided.
    For example, if the user types 'not_a_valid_id'.
    """
    pass

# Custom error for network or API problems
class ArxivFetchError(Exception):
    """
    Raised when there's an error fetching from arXiv API.
    This could be a network error, a 5xx server error, or a 404 not found.
    """
    def __init__(self, message: str, status_code: Optional[int] = None, arxiv_id: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.arxiv_id = arxiv_id
        super().__init__(self.message)

# Helper function to check if an arXiv ID is valid
# You can edit this if you want to support more ID formats
# For now, it just checks for basic alphanumeric/dot/slash

def _validate_arxiv_id(arxiv_id: str) -> str:
    """
    Validate and normalize arXiv ID.
    Removes common prefixes and checks for valid characters.
    Raises InvalidArxivIDError if the ID is not valid.
    """
    arxiv_id = arxiv_id.strip()
    if arxiv_id.startswith("arxiv:"):
        arxiv_id = arxiv_id[6:]
    if arxiv_id.startswith("abs/"):
        arxiv_id = arxiv_id[4:]
    # Only allow alphanumeric, dots, and slashes
    if not arxiv_id.replace(".", "").replace("/", "").isalnum():
        raise InvalidArxivIDError(f"Invalid arXiv ID format: {arxiv_id}")
    return arxiv_id

# Helper function to make HTTP requests with retry/backoff
# This is useful for handling temporary network/server issues
# You can adjust max_retries and base_delay if you want

def _make_request_with_retry(url: str, max_retries: int = 3, base_delay: float = 1.0) -> requests.Response:
    """
    Make HTTP request with exponential backoff retry on 5xx errors.
    Tries up to max_retries+1 times, waiting longer each time.
    Raises ArxivFetchError if all attempts fail.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=30)
            # If successful or 4xx error, return immediately
            if response.status_code < 500:
                return response
            # For 5xx errors, retry with exponential backoff
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Server error {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise ArxivFetchError(
                    f"Server error after {max_retries} retries: {response.status_code}",
                    status_code=response.status_code
                )
        except requests.RequestException as e:
            # Handles network errors (e.g., no internet)
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request failed: {e}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise ArxivFetchError(f"Request failed after {max_retries} retries: {e}")
    # Should never reach here, but just in case
    raise ArxivFetchError("Unknown error in _make_request_with_retry")

# Main function to fetch arXiv paper metadata (and optionally download PDF)
# This is the function you'll use from other Python code
# You can change the arguments or return type if you want to extend it

def fetch_arxiv_paper(
    arxiv_id: str, 
    *, 
    download_pdf: bool = False, 
    dest_dir: Optional[Path] = None
) -> PaperMetadata:
    """
    Fetch paper metadata and optionally download PDF from arXiv.
    
    Args:
        arxiv_id: arXiv identifier (e.g., '1804.02464')
        download_pdf: Whether to download the PDF (default: False)
        dest_dir: Directory to save PDF (defaults to current directory)
    Returns:
        PaperMetadata object with paper information
    Raises:
        InvalidArxivIDError: If the arXiv ID is invalid
        ArxivFetchError: If there's an error fetching from arXiv
    """
    # Validate and normalize arXiv ID
    arxiv_id = _validate_arxiv_id(arxiv_id)
    # Build the arXiv API URL
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    # Fetch metadata from arXiv
    logger.info(f"Fetching metadata for arXiv:{arxiv_id}")
    response = _make_request_with_retry(api_url)
    # Handle HTTP errors
    if response.status_code == 404:
        raise ArxivFetchError(f"Paper not found: {arxiv_id}", status_code=404, arxiv_id=arxiv_id)
    elif response.status_code != 200:
        raise ArxivFetchError(
            f"Failed to fetch metadata: {response.status_code}",
            status_code=response.status_code,
            arxiv_id=arxiv_id
        )
    # Parse the Atom XML response using feedparser
    feed = feedparser.parse(response.content)
    if not feed.entries:
        raise ArxivFetchError(f"No paper found with ID: {arxiv_id}", arxiv_id=arxiv_id)
    entry = feed.entries[0]  # There should only be one entry for a single ID
    # Extract metadata fields from the entry
    title = entry.title.strip()
    abstract = entry.summary.strip()
    # Authors are a list of dicts with a 'name' field
    authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else []
    # Categories are in entry.tags; primary_category is the one with the arXiv scheme
    categories = []
    primary_category = ""
    if hasattr(entry, 'tags'):
        for tag in entry.tags:
            if tag.term:
                categories.append(tag.term)
                if tag.get('scheme') == 'http://arxiv.org/schemas/atom':
                    primary_category = tag.term
    # Published date is an ISO string; convert to datetime
    published = datetime.fromisoformat(entry.published.replace('Z', '+00:00'))
    # DOI is optional
    doi = getattr(entry, 'arxiv_doi', None)
    # Build the PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # Create the metadata object
    metadata = PaperMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        primary_category=primary_category,
        categories=categories,
        published=published,
        doi=doi,
        pdf_url=pdf_url
    )
    # If requested, download the PDF
    if download_pdf:
        # Use the provided directory or current directory
        if dest_dir is None:
            dest_dir = Path.cwd()
        dest_dir.mkdir(parents=True, exist_ok=True)  # Make sure directory exists
        pdf_filename = f"arxiv_{arxiv_id}.pdf"
        pdf_path = dest_dir / pdf_filename
        logger.info(f"Downloading PDF to {pdf_path}")
        try:
            pdf_response = _make_request_with_retry(pdf_url)
            pdf_response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.content)
            # Check that the file is not empty
            if pdf_path.stat().st_size == 0:
                raise ArxivFetchError("Downloaded PDF is empty", arxiv_id=arxiv_id)
            metadata.pdf_path = pdf_path
            logger.info(f"PDF downloaded successfully: {pdf_path}")
        except Exception as e:
            # If something goes wrong, delete the partial file
            if pdf_path.exists():
                pdf_path.unlink()
            raise ArxivFetchError(f"Failed to download PDF: {e}", arxiv_id=arxiv_id)
    return metadata

# Command-line interface (CLI) entry point
# This lets you run the script from the terminal
# You can add more arguments if you want to extend the CLI

def main():
    """
    CLI entry point for arXiv paper ingestion.
    Parses command-line arguments and prints the result as JSON.
    """
    parser = argparse.ArgumentParser(description="Fetch arXiv paper metadata and optionally download PDF")
    parser.add_argument("arxiv_id", help="arXiv identifier (e.g., '1804.02464')")
    parser.add_argument("--pdf", action="store_true", help="Download PDF")
    parser.add_argument("--out", type=Path, help="Output directory for PDF (default: current directory)")
    args = parser.parse_args()
    try:
        metadata = fetch_arxiv_paper(
            arxiv_id=args.arxiv_id,
            download_pdf=args.pdf,
            dest_dir=args.out
        )
        # Print the metadata as pretty JSON
        print(json.dumps(metadata.model_dump(), indent=2, default=str))
    except (InvalidArxivIDError, ArxivFetchError) as e:
        # Print errors to stderr and exit with code 1
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

# This block makes sure the CLI runs only if you run this file directly
if __name__ == "__main__":
    main() 
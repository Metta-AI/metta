/**
 * URL utility functions for parsing, cleaning, and validating URLs
 */

/**
 * Remove arXiv URLs from content string
 * Handles various arXiv URL formats (abs, pdf, http, https)
 *
 * @param content - Text content that may contain arXiv URLs
 * @returns Content with arXiv URLs removed and trailing whitespace trimmed
 *
 * @example
 * cleanArxivUrls("Check this paper https://arxiv.org/abs/2204.11674")
 * // "Check this paper"
 */
export function cleanArxivUrls(content: string): string {
  const arxivPattern =
    /https?:\/\/(www\.)?(arxiv\.org\/abs\/|arxiv\.org\/pdf\/)[^\s]+/gi;
  return content.replace(arxivPattern, "").replace(/\s+$/, "");
}

/**
 * Validate if a string is a valid URL
 *
 * @param url - String to validate
 * @returns true if valid URL, false otherwise
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Extract arXiv ID from various arXiv URL formats
 *
 * @param url - arXiv URL (abs or pdf)
 * @returns arXiv ID or null if not found
 *
 * @example
 * extractArxivId("https://arxiv.org/abs/2204.11674") // "2204.11674"
 * extractArxivId("https://arxiv.org/pdf/2204.11674.pdf") // "2204.11674"
 */
export function extractArxivId(url: string): string | null {
  const match = url.match(
    /arxiv\.org\/(?:abs|pdf)\/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)/i
  );
  return match ? match[1] : null;
}

/**
 * Build arXiv abstract URL from ID
 *
 * @param arxivId - arXiv paper ID
 * @returns Full arXiv abstract URL
 */
export function buildArxivUrl(arxivId: string): string {
  return `https://arxiv.org/abs/${arxivId}`;
}

/**
 * Build arXiv PDF URL from ID
 *
 * @param arxivId - arXiv paper ID
 * @returns Full arXiv PDF URL
 */
export function buildArxivPdfUrl(arxivId: string): string {
  return `https://arxiv.org/pdf/${arxivId}.pdf`;
}

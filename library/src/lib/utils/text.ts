export function getInitialsFromName(name: string, maxLength = 3): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => word[0]?.toUpperCase() ?? "")
    .join("")
    .slice(0, Math.max(1, maxLength));
}

export function truncateText(
  text: string,
  maxLength: number,
  options: { suffix?: string } = {}
): string {
  const { suffix = "…" } = options;

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, Math.max(0, maxLength - suffix.length))}${suffix}`;
}

/**
 * Clean post content by removing URLs and extra whitespace
 * Primarily used for display purposes to remove redundant information
 *
 * @param content - Raw post content
 * @returns Cleaned content
 */
export function cleanPostContent(content: string): string {
  // Import URL cleaning utilities
  const arxivPattern =
    /https?:\/\/(www\.)?(arxiv\.org\/abs\/|arxiv\.org\/pdf\/)[^\s]+/gi;
  return content.replace(arxivPattern, "").replace(/\s+$/, "");
}

/**
 * Extract the first N words from text
 *
 * @param text - Text to extract from
 * @param wordCount - Number of words to extract
 * @returns First N words
 */
export function getFirstWords(text: string, wordCount: number): string {
  return text.split(/\s+/).slice(0, wordCount).join(" ");
}

/**
 * Sanitize text for safe HTML rendering (basic escaping)
 *
 * @param text - Text to sanitize
 * @returns Sanitized text
 */
export function sanitizeText(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Check if text contains any content after trimming
 *
 * @param text - Text to check
 * @returns true if text has content
 */
export function hasContent(text: string | null | undefined): boolean {
  return Boolean(text?.trim());
}

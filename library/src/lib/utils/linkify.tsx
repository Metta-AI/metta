import React from "react";

/**
 * Utility to convert URLs in text to clickable links
 *
 * @param text - The text content that may contain URLs
 * @returns JSX with URLs converted to clickable links
 */
export function linkifyText(text: string): React.ReactNode {
  // Regular expression to match URLs (including arXiv URLs)
  const urlRegex = /(https?:\/\/[^\s]+)/g;

  const parts = text.split(urlRegex);

  return parts.map((part, index) => {
    if (urlRegex.test(part)) {
      return (
        <a
          key={index}
          href={part}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 underline transition-colors hover:text-blue-700"
        >
          {part}
        </a>
      );
    }
    return part;
  });
}

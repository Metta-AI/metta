export function formatDate(date: Date | string | null | undefined): string {
  if (!date) {
    return "Unknown";
  }

  const dateObj = typeof date === "string" ? new Date(date) : date;

  if (Number.isNaN(dateObj.getTime())) {
    return "Unknown";
  }

  return dateObj.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatRelativeDate(
  date: Date | string | null | undefined
): string {
  if (!date) {
    return "Unknown";
  }

  const dateObj = typeof date === "string" ? new Date(date) : date;

  if (Number.isNaN(dateObj.getTime())) {
    return "Unknown";
  }

  const now = new Date();
  const diff = now.getTime() - dateObj.getTime();

  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "Just now";

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;

  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} day${days === 1 ? "" : "s"} ago`;

  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks} week${weeks === 1 ? "" : "s"} ago`;

  return dateObj.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Format a date in compact relative time format (e.g., "2h", "3d", "1w")
 * Used in social feed contexts where space is limited
 *
 * @param date - Date to format
 * @returns Compact relative time string
 *
 * @example
 * formatRelativeTimeCompact(new Date()) // "now"
 * formatRelativeTimeCompact(threeHoursAgo) // "3h"
 * formatRelativeTimeCompact(twoDaysAgo) // "2d"
 */
export function formatRelativeTimeCompact(
  date: Date | string | null | undefined
): string {
  if (!date) {
    return "Unknown";
  }

  const dateObj = typeof date === "string" ? new Date(date) : date;

  if (Number.isNaN(dateObj.getTime())) {
    return "Unknown";
  }

  const now = new Date();
  const diffInHours = Math.floor(
    (now.getTime() - dateObj.getTime()) / (1000 * 60 * 60)
  );

  if (diffInHours < 1) return "now";
  if (diffInHours < 24) return `${diffInHours}h`;

  const diffInDays = Math.floor(diffInHours / 24);
  if (diffInDays < 7) return `${diffInDays}d`;

  const diffInWeeks = Math.floor(diffInDays / 7);
  if (diffInWeeks < 4) return `${diffInWeeks}w`;

  const diffInMonths = Math.floor(diffInDays / 30);
  return `${diffInMonths}m`;
}

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

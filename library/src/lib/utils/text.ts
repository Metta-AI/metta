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

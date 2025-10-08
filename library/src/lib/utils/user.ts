/**
 * User utility functions
 */

/**
 * Generate user initials from name or email
 * Returns first 2 characters of name initials, or first character of email, or "?"
 *
 * @param name - User's full name
 * @param email - User's email address
 * @returns Initials string (1-2 uppercase characters)
 *
 * @example
 * getUserInitials("John Doe", null) // "JD"
 * getUserInitials(null, "jane@example.com") // "J"
 * getUserInitials(null, null) // "?"
 */
export function getUserInitials(
  name: string | null,
  email: string | null
): string {
  if (name) {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  }
  if (email) {
    return email.charAt(0).toUpperCase();
  }
  return "?";
}

/**
 * Get a user's display name for UI rendering
 * Priority: name > email username > "Unknown User"
 *
 * @param name - User's full name
 * @param email - User's email address
 * @returns User-friendly display name
 *
 * @example
 * getUserDisplayName("John Doe", null) // "John Doe"
 * getUserDisplayName(null, "jane@example.com") // "jane"
 * getUserDisplayName(null, null) // "Unknown User"
 */
export function getUserDisplayName(
  name: string | null | undefined,
  email: string | null | undefined
): string {
  if (name) return name;
  if (email) return email.split("@")[0];
  return "Unknown User";
}

/**
 * Type guard to check if a user object has required display information
 */
export function hasUserDisplayInfo(user: {
  name?: string | null;
  email?: string | null;
}): boolean {
  return Boolean(user.name || user.email);
}

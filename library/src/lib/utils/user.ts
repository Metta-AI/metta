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

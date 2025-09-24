/**
 * Mention parsing and resolution utilities
 *
 * Supports:
 * - @username - individual user mentions
 * - @/groupname - group within user's institution
 * - @domain.com/groupname - group in specific institution
 */

export type MentionType = "user" | "group-relative" | "group-absolute";

export interface ParsedMention {
  type: MentionType;
  raw: string;
  value: string;
  domain?: string; // For absolute group mentions
  groupName?: string; // For group mentions
  username?: string; // For user mentions
}

/**
 * Regular expressions for different mention patterns
 */
const MENTION_PATTERNS = {
  // @username (individual user)
  user: /@([a-zA-Z0-9._-]+)(?!\S)/g,

  // @/groupname (relative group in user's institution)
  groupRelative: /@\/([a-zA-Z0-9_-]+)(?!\S)/g,

  // @domain.com/groupname (absolute group in specific institution)
  groupAbsolute: /@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\/([a-zA-Z0-9_-]+)(?!\S)/g,
};

/**
 * Parse all mentions from text
 */
export function parseMentions(text: string): ParsedMention[] {
  const mentions: ParsedMention[] = [];

  // Find user mentions
  let match;
  while ((match = MENTION_PATTERNS.user.exec(text)) !== null) {
    const raw = match[0];
    const username = match[1];

    // Skip if this is part of a group mention
    const beforeMatch = text.slice(0, match.index);
    if (beforeMatch.endsWith("/") || raw.includes("/")) {
      continue;
    }

    mentions.push({
      type: "user",
      raw,
      value: username,
      username,
    });
  }

  // Reset regex state
  MENTION_PATTERNS.user.lastIndex = 0;

  // Find relative group mentions (@/groupname)
  while ((match = MENTION_PATTERNS.groupRelative.exec(text)) !== null) {
    const raw = match[0];
    const groupName = match[1];

    mentions.push({
      type: "group-relative",
      raw,
      value: groupName,
      groupName,
    });
  }

  // Reset regex state
  MENTION_PATTERNS.groupRelative.lastIndex = 0;

  // Find absolute group mentions (@domain.com/groupname)
  while ((match = MENTION_PATTERNS.groupAbsolute.exec(text)) !== null) {
    const raw = match[0];
    const domain = match[1];
    const groupName = match[2];

    mentions.push({
      type: "group-absolute",
      raw,
      value: `${domain}/${groupName}`,
      domain,
      groupName,
    });
  }

  // Reset regex state
  MENTION_PATTERNS.groupAbsolute.lastIndex = 0;

  return mentions;
}

/**
 * Get the mention pattern that matches at a specific position in text
 */
export function getMentionAtPosition(
  text: string,
  position: number
): {
  match: string;
  start: number;
  end: number;
  type: MentionType;
} | null {
  // Look backwards from position to find start of mention
  let start = position;
  while (start > 0 && text[start - 1] !== " " && text[start - 1] !== "\n") {
    start--;
  }

  // Check if we found a mention starting with @
  if (start >= text.length || text[start] !== "@") {
    return null;
  }

  // Find end of mention
  let end = position;
  while (end < text.length && text[end] !== " " && text[end] !== "\n") {
    end++;
  }

  const mentionText = text.slice(start, end);

  // Determine mention type
  if (mentionText.startsWith("@/")) {
    return {
      match: mentionText,
      start,
      end,
      type: "group-relative",
    };
  } else if (
    mentionText.includes("/") &&
    mentionText.match(/@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\//)
  ) {
    return {
      match: mentionText,
      start,
      end,
      type: "group-absolute",
    };
  } else {
    return {
      match: mentionText,
      start,
      end,
      type: "user",
    };
  }
}

/**
 * Extract query from partial mention for autocomplete
 */
export function extractMentionQuery(mentionText: string): {
  query: string;
  type: MentionType;
  domain?: string;
} {
  if (mentionText.startsWith("@/")) {
    // Relative group mention
    const query = mentionText.slice(2); // Remove @/
    return { query, type: "group-relative" };
  } else if (mentionText.includes("/")) {
    // Absolute group mention
    const match = mentionText.match(/@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\/(.*)$/);
    if (match) {
      const domain = match[1];
      const query = match[2];
      return { query, type: "group-absolute", domain };
    }
  }

  // User mention (default)
  const query = mentionText.slice(1); // Remove @
  return { query, type: "user" };
}

/**
 * Replace mentions in text with styled components or resolved users
 */
export function replaceMentionsInText(
  text: string,
  replacer: (mention: ParsedMention) => string
): string {
  const mentions = parseMentions(text);
  let result = text;

  // Replace in reverse order to maintain positions
  mentions.reverse().forEach((mention) => {
    const replacement = replacer(mention);
    result = result.replace(mention.raw, replacement);
  });

  return result;
}

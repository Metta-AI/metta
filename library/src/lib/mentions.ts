/**
 * Mention parsing and resolution utilities
 *
 * Supports:
 * - @username - individual user mentions
 * - @institution-name - institution mentions
 * - @/groupname - group within user's institution
 * - @domain.com/groupname - group in specific institution by domain
 * - @institution-name/groupname - group in specific institution by name
 */

export type MentionType =
  | "user"
  | "institution"
  | "group-relative"
  | "group-absolute"
  | "group-institution";

export interface ParsedMention {
  type: MentionType;
  raw: string;
  value: string;
  domain?: string; // For absolute group mentions
  groupName?: string; // For group mentions
  username?: string; // For user mentions
  institutionName?: string; // For institution mentions
}

/**
 * Regular expressions for different mention patterns
 */
const MENTION_PATTERNS = {
  // @/groupname (relative group in user's institution)
  groupRelative: /@\/([a-zA-Z0-9_-]+)(?!\S)/g,

  // @domain.com/groupname (absolute group in specific institution by domain)
  groupAbsolute: /@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\/([a-zA-Z0-9_-]+)(?!\S)/g,

  // @institution-name/groupname (absolute group in specific institution by name)
  groupInstitution: /@([a-zA-Z0-9._-]+)\/([a-zA-Z0-9_-]+)(?!\S)/g,

  // @institution-name (institution mentions)
  institution: /@([a-zA-Z0-9._-]+)(?!\S)/g,

  // @username (individual user)
  user: /@([a-zA-Z0-9._-]+)(?!\S)/g,
};

/**
 * Parse all mentions from text
 */
export function parseMentions(text: string): ParsedMention[] {
  const mentions: ParsedMention[] = [];
  const processedPositions = new Set<number>();

  let match;

  // 1. Find relative group mentions (@/groupname) - highest priority
  while ((match = MENTION_PATTERNS.groupRelative.exec(text)) !== null) {
    const raw = match[0];
    const groupName = match[1];
    const startPos = match.index!;
    const endPos = startPos + raw.length;

    // Mark this position range as processed
    for (let i = startPos; i < endPos; i++) {
      processedPositions.add(i);
    }

    mentions.push({
      type: "group-relative",
      raw,
      value: groupName,
      groupName,
    });
  }

  // Reset regex state
  MENTION_PATTERNS.groupRelative.lastIndex = 0;

  // 2. Find absolute group mentions by domain (@domain.com/groupname)
  while ((match = MENTION_PATTERNS.groupAbsolute.exec(text)) !== null) {
    const raw = match[0];
    const domain = match[1];
    const groupName = match[2];
    const startPos = match.index!;
    const endPos = startPos + raw.length;

    // Skip if already processed
    let alreadyProcessed = false;
    for (let i = startPos; i < endPos; i++) {
      if (processedPositions.has(i)) {
        alreadyProcessed = true;
        break;
      }
    }

    if (!alreadyProcessed) {
      // Mark this position range as processed
      for (let i = startPos; i < endPos; i++) {
        processedPositions.add(i);
      }

      mentions.push({
        type: "group-absolute",
        raw,
        value: `${domain}/${groupName}`,
        domain,
        groupName,
      });
    }
  }

  // Reset regex state
  MENTION_PATTERNS.groupAbsolute.lastIndex = 0;

  // 3. Find institution-based group mentions (@institution-name/groupname)
  while ((match = MENTION_PATTERNS.groupInstitution.exec(text)) !== null) {
    const raw = match[0];
    const institutionName = match[1];
    const groupName = match[2];
    const startPos = match.index!;
    const endPos = startPos + raw.length;

    // Skip if already processed by domain pattern or if institutionName looks like a domain
    let alreadyProcessed = false;
    for (let i = startPos; i < endPos; i++) {
      if (processedPositions.has(i)) {
        alreadyProcessed = true;
        break;
      }
    }

    // Skip if this looks like a domain (contains dots and TLD)
    if (
      institutionName.includes(".") &&
      institutionName.match(/\.[a-zA-Z]{2,}$/)
    ) {
      alreadyProcessed = true;
    }

    if (!alreadyProcessed) {
      // Mark this position range as processed
      for (let i = startPos; i < endPos; i++) {
        processedPositions.add(i);
      }

      mentions.push({
        type: "group-institution",
        raw,
        value: `${institutionName}/${groupName}`,
        institutionName,
        groupName,
      });
    }
  }

  // Reset regex state
  MENTION_PATTERNS.groupInstitution.lastIndex = 0;

  // 4. Find institution mentions (@institution-name)
  while ((match = MENTION_PATTERNS.institution.exec(text)) !== null) {
    const raw = match[0];
    const institutionName = match[1];
    const startPos = match.index!;
    const endPos = startPos + raw.length;

    // Skip if already processed
    let alreadyProcessed = false;
    for (let i = startPos; i < endPos; i++) {
      if (processedPositions.has(i)) {
        alreadyProcessed = true;
        break;
      }
    }

    if (!alreadyProcessed) {
      // Mark this position range as processed
      for (let i = startPos; i < endPos; i++) {
        processedPositions.add(i);
      }

      mentions.push({
        type: "institution",
        raw,
        value: institutionName,
        institutionName,
      });
    }
  }

  // Reset regex state
  MENTION_PATTERNS.institution.lastIndex = 0;

  // 5. Find user mentions (@username) - lowest priority
  while ((match = MENTION_PATTERNS.user.exec(text)) !== null) {
    const raw = match[0];
    const username = match[1];
    const startPos = match.index!;
    const endPos = startPos + raw.length;

    // Skip if already processed
    let alreadyProcessed = false;
    for (let i = startPos; i < endPos; i++) {
      if (processedPositions.has(i)) {
        alreadyProcessed = true;
        break;
      }
    }

    if (!alreadyProcessed) {
      mentions.push({
        type: "user",
        raw,
        value: username,
        username,
      });
    }
  }

  // Reset regex state
  MENTION_PATTERNS.user.lastIndex = 0;

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

  // Determine mention type by order of specificity
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
  } else if (
    mentionText.includes("/") &&
    !mentionText.match(/@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\//)
  ) {
    return {
      match: mentionText,
      start,
      end,
      type: "group-institution",
    };
  } else {
    return {
      match: mentionText,
      start,
      end,
      type: "user", // Could also be institution - resolved during search
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
  institutionName?: string;
} {
  if (mentionText.startsWith("@/")) {
    // Relative group mention
    const query = mentionText.slice(2); // Remove @/
    return { query, type: "group-relative" };
  } else if (mentionText.includes("/")) {
    // Check if it's a domain-based group mention first
    const domainMatch = mentionText.match(
      /@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\/(.*)$/
    );
    if (domainMatch) {
      const domain = domainMatch[1];
      const query = domainMatch[2];
      return { query, type: "group-absolute", domain };
    }

    // Otherwise it's an institution-name based group mention
    const institutionMatch = mentionText.match(/@([a-zA-Z0-9._-]+)\/(.*)$/);
    if (institutionMatch) {
      const institutionName = institutionMatch[1];
      const query = institutionMatch[2];
      return { query, type: "group-institution", institutionName };
    }
  }

  // User mention (could also be institution - will be determined by search)
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

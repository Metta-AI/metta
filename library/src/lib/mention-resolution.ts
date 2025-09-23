/**
 * Mention resolution utilities
 *
 * Resolves different types of mentions to actual user IDs for notifications
 */

import { prisma } from "@/lib/db/prisma";
import { ParsedMention, parseMentions } from "./mentions";

export interface ResolvedMention {
  type: "user" | "group";
  originalMention: string;
  userIds: string[];
  groupName?: string;
  institutionName?: string;
}

/**
 * Resolve all mentions from a list of mention strings to user IDs
 */
export async function resolveMentions(
  mentionStrings: string[],
  currentUserId: string
): Promise<ResolvedMention[]> {
  const resolved: ResolvedMention[] = [];

  // Parse all mentions from the strings
  const allParsedMentions: ParsedMention[] = [];
  for (const mentionString of mentionStrings) {
    const parsed = parseMentions(mentionString);
    allParsedMentions.push(...parsed);
  }

  for (const mention of allParsedMentions) {
    try {
      const resolution = await resolveSingleMention(mention, currentUserId);
      if (resolution && resolution.userIds.length > 0) {
        resolved.push(resolution);
      }
    } catch (error) {
      console.error(`Failed to resolve mention: ${mention.raw}`, error);
    }
  }

  return resolved;
}

/**
 * Resolve a single parsed mention to user IDs
 */
async function resolveSingleMention(
  mention: ParsedMention,
  currentUserId: string
): Promise<ResolvedMention | null> {
  switch (mention.type) {
    case "user":
      return resolveUserMention(mention);

    case "group-relative":
      return resolveRelativeGroupMention(mention, currentUserId);

    case "group-absolute":
      return resolveAbsoluteGroupMention(mention);

    default:
      return null;
  }
}

/**
 * Resolve @username mentions
 */
async function resolveUserMention(
  mention: ParsedMention
): Promise<ResolvedMention | null> {
  if (!mention.username) return null;

  // Find user by email prefix (since usernames are typically email prefixes)
  const user = await prisma.user.findFirst({
    where: {
      OR: [
        { email: { startsWith: mention.username + "@" } },
        { name: { equals: mention.username, mode: "insensitive" } },
      ],
    },
    select: { id: true },
  });

  if (!user) return null;

  return {
    type: "user",
    originalMention: mention.raw,
    userIds: [user.id],
  };
}

/**
 * Resolve @/groupname mentions (relative to user's institutions)
 */
async function resolveRelativeGroupMention(
  mention: ParsedMention,
  currentUserId: string
): Promise<ResolvedMention | null> {
  if (!mention.groupName) return null;

  // Get user's institutions
  const userInstitutions = await prisma.userInstitution.findMany({
    where: {
      userId: currentUserId,
      isActive: true,
    },
    select: { institutionId: true },
  });

  const institutionIds = userInstitutions.map((ui) => ui.institutionId);
  if (institutionIds.length === 0) return null;

  // Find groups with this name in user's institutions
  const groups = await prisma.group.findMany({
    where: {
      name: mention.groupName,
      institutionId: { in: institutionIds },
    },
    include: {
      userGroups: {
        where: { isActive: true },
        include: { user: { select: { id: true } } },
      },
      institution: {
        select: { name: true },
      },
    },
  });

  if (groups.length === 0) return null;

  // Get all unique user IDs from all matching groups
  const userIds = new Set<string>();
  let institutionName = "";

  for (const group of groups) {
    institutionName = group.institution.name;
    for (const userGroup of group.userGroups) {
      userIds.add(userGroup.user.id);
    }
  }

  return {
    type: "group",
    originalMention: mention.raw,
    userIds: Array.from(userIds),
    groupName: mention.groupName,
    institutionName,
  };
}

/**
 * Resolve @domain.com/groupname mentions (absolute group references)
 */
async function resolveAbsoluteGroupMention(
  mention: ParsedMention
): Promise<ResolvedMention | null> {
  if (!mention.domain || !mention.groupName) return null;

  // Find institution by domain
  const institution = await prisma.institution.findUnique({
    where: { domain: mention.domain },
    select: { id: true, name: true },
  });

  if (!institution) return null;

  // Find group in the specific institution
  const group = await prisma.group.findUnique({
    where: {
      name_institutionId: {
        name: mention.groupName,
        institutionId: institution.id,
      },
    },
    include: {
      userGroups: {
        where: { isActive: true },
        include: { user: { select: { id: true } } },
      },
    },
  });

  if (!group) return null;

  const userIds = group.userGroups.map((ug) => ug.user.id);

  return {
    type: "group",
    originalMention: mention.raw,
    userIds,
    groupName: mention.groupName,
    institutionName: institution.name,
  };
}

/**
 * Get all unique user IDs from resolved mentions (excluding the current user)
 */
export function extractUserIdsFromResolution(
  resolvedMentions: ResolvedMention[],
  excludeUserId?: string
): string[] {
  const userIds = new Set<string>();

  for (const resolution of resolvedMentions) {
    for (const userId of resolution.userIds) {
      if (userId !== excludeUserId) {
        userIds.add(userId);
      }
    }
  }

  return Array.from(userIds);
}


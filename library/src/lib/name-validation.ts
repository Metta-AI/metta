import { prisma } from "@/lib/db/prisma";
import { ActionError } from "@/lib/actionClient";

/**
 * Validates that a name doesn't conflict with existing identifiers across the system
 * This ensures unique mentions and tags work correctly
 */
export async function validateNameUniqueness(
  name: string,
  type: "institution" | "author" | "group",
  excludeId?: string
): Promise<{ isValid: boolean; conflictType?: string; message?: string }> {
  if (!name || name.trim() === "") {
    return {
      isValid: false,
      message: "Name cannot be empty",
    };
  }

  const normalizedName = name.trim();

  // Check against institutions
  if (type !== "institution") {
    const conflictingInstitution = await prisma.institution.findUnique({
      where: { name: normalizedName },
    });

    if (conflictingInstitution && conflictingInstitution.id !== excludeId) {
      return {
        isValid: false,
        conflictType: "institution",
        message:
          "This name conflicts with an existing institution. Please choose a different name to ensure unique mentions.",
      };
    }
  }

  // Check against author usernames
  if (type !== "author") {
    const conflictingAuthor = await prisma.author.findUnique({
      where: { username: normalizedName },
    });

    if (conflictingAuthor && conflictingAuthor.id !== excludeId) {
      return {
        isValid: false,
        conflictType: "author",
        message:
          "This name conflicts with an existing author username. Please choose a different name to ensure unique mentions.",
      };
    }
  }

  // Check against user names (for @username mentions)
  const conflictingUserByName = await prisma.user.findFirst({
    where: { name: { equals: normalizedName, mode: "insensitive" } },
  });

  if (conflictingUserByName) {
    return {
      isValid: false,
      conflictType: "user_name",
      message:
        "This name conflicts with an existing user name. Please choose a different name to ensure unique mentions.",
    };
  }

  // Check against user email prefixes (for @username mentions)
  const conflictingUserByEmail = await prisma.user.findFirst({
    where: { email: { startsWith: normalizedName + "@" } },
  });

  if (conflictingUserByEmail) {
    return {
      isValid: false,
      conflictType: "user_email",
      message:
        "This name conflicts with an existing user's email prefix. Please choose a different name to ensure unique mentions.",
    };
  }

  // Check against group names
  if (type !== "group") {
    const conflictingGroup = await prisma.group.findFirst({
      where: {
        name: normalizedName,
        ...(excludeId ? { NOT: { id: excludeId } } : {}),
      },
    });

    if (conflictingGroup) {
      return {
        isValid: false,
        conflictType: "group",
        message:
          "This name conflicts with an existing group name. Please choose a different name to ensure unique mentions.",
      };
    }
  }

  return { isValid: true };
}

/**
 * Validates author username uniqueness across all entity types
 */
export async function validateAuthorUsername(
  username: string,
  excludeAuthorId?: string
): Promise<void> {
  const result = await validateNameUniqueness(
    username,
    "author",
    excludeAuthorId
  );

  if (!result.isValid) {
    throw new ActionError(result.message);
  }
}

/**
 * Validates institution name uniqueness across all entity types
 */
export async function validateInstitutionName(
  name: string,
  excludeInstitutionId?: string
): Promise<void> {
  const result = await validateNameUniqueness(
    name,
    "institution",
    excludeInstitutionId
  );

  if (!result.isValid) {
    throw new ActionError(result.message);
  }
}

/**
 * Validates group name uniqueness across all entity types
 */
export async function validateGroupName(
  name: string,
  excludeGroupId?: string
): Promise<void> {
  const result = await validateNameUniqueness(name, "group", excludeGroupId);

  if (!result.isValid) {
    throw new ActionError(result.message);
  }
}

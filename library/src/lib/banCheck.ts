import "server-only";

import { prisma } from "@/lib/db/prisma";
import { ActionError } from "@/lib/actionClient";

/**
 * Check if a user is banned and throw an error if they are
 */
export async function checkUserNotBanned(userId: string): Promise<void> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      isBanned: true,
      banReason: true,
      bannedAt: true,
    },
  });

  if (!user) {
    throw new ActionError("User not found");
  }

  if (user.isBanned) {
    throw new ActionError(
      `Your account has been suspended. Reason: ${user.banReason || "No reason provided"}. Please contact support if you believe this is a mistake.`
    );
  }
}

/**
 * Check if a user is banned (returns boolean instead of throwing)
 */
export async function isUserBanned(userId: string): Promise<boolean> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      isBanned: true,
    },
  });

  return user?.isBanned ?? false;
}

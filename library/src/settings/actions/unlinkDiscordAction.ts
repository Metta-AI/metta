"use server";

import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

export const unlinkDiscordAction = actionClient.action(async () => {
  const session = await getSessionOrRedirect();
  const userId = session.user.id;

  // Get current Discord link info
  const existingPreferences = await prisma.notificationPreference.findMany({
    where: {
      userId,
      discordUserId: { not: null },
    },
    select: {
      discordUserId: true,
      discordUsername: true,
    },
  });

  if (existingPreferences.length === 0) {
    throw new Error("No Discord account linked");
  }

  const discordInfo = existingPreferences[0];

  // Remove Discord linking from all notification preferences
  await prisma.notificationPreference.updateMany({
    where: {
      userId,
      discordUserId: discordInfo.discordUserId,
    },
    data: {
      discordEnabled: false,
      discordUserId: null,
      discordUsername: null,
      discordLinkedAt: null,
    },
  });

  return { success: true };
});

"use server";

import { z } from "zod/v4";
import { getSessionOrRedirect } from "@/lib/auth";
import {
  markNotificationsRead,
  markAllNotificationsRead,
} from "@/lib/notifications";
import { prisma } from "@/lib/db/prisma";

export type MarkNotificationsReadInput = {
  notificationIds?: string[];
  markAllRead?: boolean;
};

const markReadSchema = z.object({
  notificationIds: z.array(z.string()).optional(),
  markAllRead: z.boolean().default(false),
});

/**
 * Server action to mark notifications as read
 */
export async function markNotificationsReadAction(
  input: MarkNotificationsReadInput
): Promise<{ success: boolean }> {
  const session = await getSessionOrRedirect();
  const { notificationIds, markAllRead } = markReadSchema.parse(input);

  if (markAllRead) {
    // Mark all notifications as read
    await markAllNotificationsRead(session.user.id);
  } else if (notificationIds && notificationIds.length > 0) {
    // Mark specific notifications as read
    // First verify these notifications belong to the current user
    const userNotifications = await prisma.notification.findMany({
      where: {
        id: { in: notificationIds },
        userId: session.user.id, // Security: only mark user's own notifications
      },
      select: { id: true },
    });

    const verifiedIds = userNotifications.map((n) => n.id);
    if (verifiedIds.length > 0) {
      await markNotificationsRead(verifiedIds);
    }
  }

  return { success: true };
}

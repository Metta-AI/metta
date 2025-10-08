"use server";

import { zfd } from "zod-form-data";
import { actionClient } from "@/lib/actionClient";
import { getSessionOrRedirect } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";

const inputSchema = zfd.formData({
  notificationIds: zfd.text().optional(),
  markAllRead: zfd.text().optional(),
});

export const markNotificationsReadAction = actionClient
  .inputSchema(inputSchema)
  .action(async ({ parsedInput: input }) => {
    const notificationIds = input.notificationIds
      ? JSON.parse(input.notificationIds)
      : undefined;
    const markAllRead = input.markAllRead === "true";
    const session = await getSessionOrRedirect();
    const userId = session.user.id;

    if (markAllRead) {
      // Mark all notifications as read
      await prisma.notification.updateMany({
        where: {
          userId,
          isRead: false,
        },
        data: {
          isRead: true,
        },
      });
    } else if (notificationIds && notificationIds.length > 0) {
      // Mark specific notifications as read
      await prisma.notification.updateMany({
        where: {
          id: { in: notificationIds },
          userId, // Ensure user owns these notifications
        },
        data: {
          isRead: true,
        },
      });
    }

    return { success: true };
  });

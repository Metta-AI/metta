import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import {
  markNotificationsRead,
  markAllNotificationsRead,
} from "@/lib/notifications";
import { AuthenticationError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

const markReadSchema = z.object({
  notificationIds: z.array(z.string()).optional(),
  markAllRead: z.boolean().default(false),
});

export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    const body = await request.json();
    const { notificationIds, markAllRead } = markReadSchema.parse(body);

    if (markAllRead) {
      // Mark all notifications as read
      await markAllNotificationsRead(session.user.id);
    } else if (notificationIds && notificationIds.length > 0) {
      // Mark specific notifications as read
      // First verify these notifications belong to the current user
      const { prisma } = await import("@/lib/db/prisma");
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

    return NextResponse.json({ success: true });
  } catch (error) {
    return handleApiError(error, {
      endpoint: "POST /api/notifications/mark-read",
    });
  }
}

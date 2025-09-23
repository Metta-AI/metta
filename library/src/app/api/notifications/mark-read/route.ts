import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { getSessionOrRedirect } from "@/lib/auth";
import { markNotificationsRead, markAllNotificationsRead } from "@/lib/notifications";

const markReadSchema = z.object({
  notificationIds: z.array(z.string()).optional(),
  markAllRead: z.boolean().default(false),
});

export async function POST(request: NextRequest) {
  try {
    const session = await getSessionOrRedirect();
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

      const verifiedIds = userNotifications.map(n => n.id);
      if (verifiedIds.length > 0) {
        await markNotificationsRead(verifiedIds);
      }
    }

    return NextResponse.json({ success: true });

  } catch (error) {
    console.error("Error marking notifications as read:", error);
    return NextResponse.json(
      { error: "Failed to mark notifications as read" },
      { status: 500 }
    );
  }
}

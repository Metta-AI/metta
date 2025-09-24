import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { getSessionOrRedirect } from "@/lib/auth";
import {
  getUserNotifications,
  getNotificationCounts,
} from "@/lib/notifications";

const querySchema = z.object({
  limit: z.coerce.number().min(1).max(50).default(20),
  offset: z.coerce.number().min(0).default(0),
  includeRead: z.coerce.boolean().default(true),
});

export async function GET(request: NextRequest) {
  try {
    const session = await getSessionOrRedirect();

    const { searchParams } = new URL(request.url);
    const params = querySchema.parse({
      limit: searchParams.get("limit"),
      offset: searchParams.get("offset"),
      includeRead: searchParams.get("includeRead"),
    });

    // Get notifications and counts in parallel
    const [notifications, counts] = await Promise.all([
      getUserNotifications(session.user.id, params),
      getNotificationCounts(session.user.id),
    ]);

    return NextResponse.json({
      notifications,
      counts,
      pagination: {
        limit: params.limit,
        offset: params.offset,
        hasMore: notifications.length === params.limit,
      },
    });
  } catch (error) {
    console.error("Error loading notifications:", error);
    return NextResponse.json(
      { error: "Failed to load notifications" },
      { status: 500 }
    );
  }
}

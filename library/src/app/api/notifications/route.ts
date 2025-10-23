import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import {
  getUserNotifications,
  getNotificationCounts,
} from "@/lib/notifications";
import { AuthenticationError } from "@/lib/errors";
import { withErrorHandler } from "@/lib/api/error-handler";

const querySchema = z.object({
  limit: z.coerce.number().min(1).max(50).default(20),
  offset: z.coerce.number().min(0).default(0),
  includeRead: z.coerce.boolean().default(true),
});

export const GET = withErrorHandler(async (request: NextRequest) => {
  const session = await auth();

  if (!isSignedIn(session)) {
    throw new AuthenticationError();
  }

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
});

import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { getNotificationCounts } from "@/lib/notifications";
import { AuthenticationError } from "@/lib/errors";
import { withErrorHandler } from "@/lib/api/error-handler";

export const GET = withErrorHandler(async (request: NextRequest) => {
  const session = await auth();

  if (!isSignedIn(session)) {
    throw new AuthenticationError();
  }

  const counts = await getNotificationCounts(session.user.id);

  return NextResponse.json(counts);
});

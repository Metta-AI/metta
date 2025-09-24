import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { getNotificationCounts } from "@/lib/notifications";

export async function GET(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    const counts = await getNotificationCounts(session.user.id);

    return NextResponse.json(counts);
  } catch (error) {
    console.error("Error getting notification counts:", error);
    return NextResponse.json(
      { error: "Failed to get notification counts" },
      { status: 500 }
    );
  }
}

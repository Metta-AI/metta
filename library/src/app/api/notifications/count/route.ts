import { NextRequest, NextResponse } from "next/server";

import { getSessionOrRedirect } from "@/lib/auth";
import { getNotificationCounts } from "@/lib/notifications";

export async function GET(request: NextRequest) {
  try {
    const session = await getSessionOrRedirect();
    
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

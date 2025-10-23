import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import {
  getUserNotificationPreferences,
  updateNotificationPreferences,
  getDeliveryStats,
} from "@/lib/notification-preferences";
import { AuthenticationError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

// Schema for updating preferences
const updatePreferencesSchema = z.object({
  preferences: z.record(
    z.string(),
    z.object({
      emailEnabled: z.boolean().optional(),
      discordEnabled: z.boolean().optional(),
    })
  ),
});

// GET /api/notification-preferences
// Returns user's notification preferences and delivery stats
export async function GET(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    const { searchParams } = new URL(request.url);
    const includeStats = searchParams.get("includeStats") === "true";

    // Get user preferences
    const userPreferences = await getUserNotificationPreferences(
      session.user.id
    );

    let stats = null;
    if (includeStats) {
      stats = await getDeliveryStats(session.user.id);
    }

    return NextResponse.json({
      preferences: userPreferences.preferences,
      ...(stats && { deliveryStats: stats }),
    });
  } catch (error) {
    return handleApiError(error, {
      endpoint: "GET /api/notification-preferences",
    });
  }
}

// PUT /api/notification-preferences
// Updates user's notification preferences
export async function PUT(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    const body = await request.json();
    const { preferences } = updatePreferencesSchema.parse(body);

    // Update preferences
    await updateNotificationPreferences(session.user.id, preferences);

    return NextResponse.json({
      message: "Preferences updated successfully",
      preferences,
    });
  } catch (error) {
    return handleApiError(error, {
      endpoint: "PUT /api/notification-preferences",
    });
  }
}

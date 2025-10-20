import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { prisma } from "@/lib/db/prisma";
import { AuthenticationError, BadRequestError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";
import { Logger } from "@/lib/logging/logger";

// GET /api/discord/link - Get Discord linking status
export async function GET() {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    // Get Discord linking status from notification preferences
    const discordPreferences = await prisma.notificationPreference.findMany({
      where: {
        userId: session.user.id,
        discordUserId: { not: null },
      },
      select: {
        discordUserId: true,
        discordUsername: true,
        discordLinkedAt: true,
        type: true,
        discordEnabled: true,
      },
    });

    if (discordPreferences.length === 0) {
      return NextResponse.json({
        isLinked: false,
        discordUsername: null,
        discordUserId: null,
        message: "Discord account not linked",
      });
    }

    // Get Discord info from first preference (they should all be the same)
    const firstPref = discordPreferences[0];

    // Count enabled notification types
    const enabledTypes = discordPreferences
      .filter((pref) => pref.discordEnabled)
      .map((pref) => pref.type);

    return NextResponse.json({
      isLinked: true,
      discordUsername: firstPref.discordUsername,
      discordUserId: firstPref.discordUserId,
      discordLinkedAt: firstPref.discordLinkedAt,
      enabledNotificationTypes: enabledTypes,
      message: `Discord account ${firstPref.discordUsername} is linked`,
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/discord/link" });
  }
}

// DELETE /api/discord/link - Unlink Discord account
export async function DELETE() {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    // Get current Discord link info
    const existingPreferences = await prisma.notificationPreference.findMany({
      where: {
        userId: session.user.id,
        discordUserId: { not: null },
      },
      select: {
        discordUserId: true,
        discordUsername: true,
      },
    });

    if (existingPreferences.length === 0) {
      throw new BadRequestError("No Discord account linked");
    }

    const discordInfo = existingPreferences[0];

    // Remove Discord linking from all notification preferences
    await prisma.notificationPreference.updateMany({
      where: {
        userId: session.user.id,
        discordUserId: discordInfo.discordUserId,
      },
      data: {
        discordEnabled: false,
        discordUserId: null,
        discordUsername: null,
        discordLinkedAt: null,
      },
    });

    Logger.info("Discord account unlinked", {
      userId: session.user.id,
      discordUsername: discordInfo.discordUsername,
    });

    return NextResponse.json({
      success: true,
      message: `Discord account ${discordInfo.discordUsername} unlinked successfully`,
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "DELETE /api/discord/link" });
  }
}

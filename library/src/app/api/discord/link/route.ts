import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { discordBot } from "@/lib/external-notifications/discord-bot";
import { prisma } from "@/lib/db/prisma";

// GET /api/discord/link - Get Discord linking status
export async function GET() {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
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
    console.error("Error getting Discord link status:", error);
    return NextResponse.json(
      {
        error: "Failed to get Discord link status",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

// DELETE /api/discord/link - Unlink Discord account
export async function DELETE() {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
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
      return NextResponse.json(
        { error: "No Discord account linked" },
        { status: 400 }
      );
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

    // Optionally send farewell DM
    try {
      if (discordInfo.discordUserId) {
        // We could send a farewell message here, but it's optional
        // await discordBot.sendFarewellMessage(discordInfo.discordUserId);
      }
    } catch (error) {
      console.warn("Failed to send farewell message:", error);
      // Don't fail the unlinking if farewell message fails
    }

    return NextResponse.json({
      success: true,
      message: `Discord account ${discordInfo.discordUsername} unlinked successfully`,
    });
  } catch (error) {
    console.error("Error unlinking Discord account:", error);
    return NextResponse.json(
      {
        error: "Failed to unlink Discord account",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

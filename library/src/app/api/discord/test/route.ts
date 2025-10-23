import { NextRequest, NextResponse } from "next/server";
import { z } from "zod/v4";

import { auth, isSignedIn } from "@/lib/auth";
import { discordBot } from "@/lib/external-notifications/discord-bot";
import type { NotificationWithDetails } from "@/lib/external-notifications/email";
import { prisma } from "@/lib/db/prisma";
import { AuthenticationError, BadRequestError } from "@/lib/errors";
import { handleApiError } from "@/lib/api/error-handler";

// Schema for test Discord message request
const testDiscordSchema = z.object({
  action: z.enum(["config", "send"]),
  testMessage: z.string().optional(),
});

// GET /api/discord/test - Check Discord bot configuration
export async function GET() {
  try {
    const config = await discordBot.getConfigurationInfo();
    const isValid = await discordBot.testConfiguration();

    return NextResponse.json({
      configuration: config,
      isValid,
      message: isValid
        ? `✅ Discord bot is ready as ${config.botUser}`
        : `❌ Discord bot configuration invalid`,
    });
  } catch (error) {
    return handleApiError(error, { endpoint: "GET /api/discord/test" });
  }
}

// POST /api/discord/test - Send test Discord message
export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      throw new AuthenticationError();
    }

    const body = await request.json();
    const { action, testMessage } = testDiscordSchema.parse(body);

    if (action === "config") {
      // Just return configuration info
      const config = await discordBot.getConfigurationInfo();
      const isValid = await discordBot.testConfiguration();

      return NextResponse.json({
        configuration: config,
        isValid,
        message: isValid
          ? `✅ Discord bot is ready as ${config.botUser}`
          : `❌ Discord bot configuration invalid`,
      });
    }

    if (action === "send") {
      // Check if user has Discord linked
      const discordPreference = await prisma.notificationPreference.findFirst({
        where: {
          userId: session.user.id,
          discordUserId: { not: null },
        },
      });

      if (!discordPreference || !discordPreference.discordUserId) {
        throw new BadRequestError("No Discord account linked");
      }

      // Create a mock notification for testing
      const mockNotification: NotificationWithDetails = {
        id: "test-discord-notification",
        userId: session.user.id,
        type: "SYSTEM" as const,
        isRead: false,
        title: "Test Discord DM",
        message:
          testMessage ||
          "This is a test Discord DM to verify your bot integration is working correctly.",
        actionUrl: null,
        createdAt: new Date(),
        updatedAt: new Date(),
        actorId: null,
        postId: null,
        commentId: null,
        mentionText: null,
        actor: null,
        user: {
          id: session.user.id,
          name: session.user.name ?? null,
          email: session.user.email ?? null,
        },
        post: null,
        comment: null,
      };

      const success = await discordBot.sendNotification(
        mockNotification,
        discordPreference.discordUserId
      );

      return NextResponse.json({
        success,
        discordUser: discordPreference.discordUsername,
        discordUserId: discordPreference.discordUserId,
        message: success
          ? `✅ Test Discord DM sent successfully to ${discordPreference.discordUsername}`
          : `❌ Failed to send test Discord DM to ${discordPreference.discordUsername}`,
      });
    }

    throw new BadRequestError("Invalid action");
  } catch (error) {
    return handleApiError(error, { endpoint: "POST /api/discord/test" });
  }
}

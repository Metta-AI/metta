import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { discordAuth, DiscordAuthService } from "@/lib/discord-auth";
import { discordBot } from "@/lib/external-notifications/discord-bot";
import { prisma } from "@/lib/db/prisma";
import type { NotificationType } from "@prisma/client";
import { Logger } from "@/lib/logging/logger";

// GET /api/discord/auth - Handle Discord OAuth callback
export async function GET(request: NextRequest) {
  try {
    const session = await auth();

    if (!isSignedIn(session)) {
      // Redirect to sign in instead of returning JSON since this is a callback
      return NextResponse.redirect(new URL("/api/auth/signin", request.url));
    }

    const { searchParams } = new URL(request.url);
    const code = searchParams.get("code");
    const error = searchParams.get("error");

    // Handle Discord OAuth errors
    if (error) {
      Logger.warn("Discord OAuth error received", {
        error,
        userId: session.user.id,
      });
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_oauth_error");
      errorRedirectUrl.searchParams.set("message", error);
      return NextResponse.redirect(errorRedirectUrl);
    }

    if (!code) {
      Logger.warn("Discord OAuth missing code", { userId: session.user.id });
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_oauth_no_code");
      errorRedirectUrl.searchParams.set(
        "message",
        "No authorization code received"
      );
      return NextResponse.redirect(errorRedirectUrl);
    }

    // Exchange code for token
    Logger.debug("Exchanging Discord code for token", {
      userId: session.user.id,
    });
    const tokenResponse = await discordAuth.exchangeCodeForToken(code);
    Logger.debug("Discord token received", { userId: session.user.id });

    // Get Discord user info
    const discordUser = await discordAuth.getDiscordUser(
      tokenResponse.access_token
    );
    Logger.info("Discord user info retrieved", {
      userId: session.user.id,
      discordUsername: discordUser.username,
      discordId: discordUser.id,
    });

    // Check if this Discord account is already linked to another user
    const existingLink = await prisma.notificationPreference.findFirst({
      where: {
        discordUserId: discordUser.id,
      },
      include: {
        user: {
          select: { id: true, name: true, email: true },
        },
      },
    });

    if (existingLink && existingLink.userId !== session.user.id) {
      Logger.warn("Discord account already linked to another user", {
        userId: session.user.id,
        discordId: discordUser.id,
        existingUserId: existingLink.userId,
      });
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_already_linked");
      errorRedirectUrl.searchParams.set(
        "message",
        "This Discord account is already linked to another user"
      );
      return NextResponse.redirect(errorRedirectUrl);
    }

    // Link Discord account to user for all notification types
    const notificationTypes: NotificationType[] = [
      "MENTION",
      "COMMENT",
      "REPLY",
      "LIKE",
      "FOLLOW",
      "PAPER_SUGGESTION",
      "SYSTEM",
    ];

    const username = DiscordAuthService.formatDiscordUsername(discordUser);
    const displayName = DiscordAuthService.getDisplayName(discordUser);

    await prisma.$transaction(async (tx) => {
      for (const type of notificationTypes) {
        await tx.notificationPreference.upsert({
          where: {
            userId_type: {
              userId: session.user.id,
              type,
            },
          },
          create: {
            userId: session.user.id,
            type,
            emailEnabled: true,
            discordEnabled: true, // Enable Discord by default when linking
            discordUserId: discordUser.id,
            discordUsername: username,
            discordLinkedAt: new Date(),
          },
          update: {
            discordEnabled: true,
            discordUserId: discordUser.id,
            discordUsername: username,
            discordLinkedAt: new Date(),
          },
        });
      }
    });

    // Send welcome message
    const welcomeSent = await discordBot.sendWelcomeMessage(
      discordUser.id,
      displayName
    );
    Logger.info("Discord welcome message sent", {
      userId: session.user.id,
      discordId: discordUser.id,
      success: welcomeSent,
    });

    // Revoke the access token (we don't need to store it since we only needed user info)
    await discordAuth.revokeToken(tokenResponse.access_token);

    // Redirect back to settings with success message
    Logger.info("Discord account linked successfully", {
      userId: session.user.id,
      discordUsername: username,
      discordId: discordUser.id,
    });
    const successRedirectUrl = new URL("/settings", request.url);
    successRedirectUrl.searchParams.set("success", "discord_linked");
    successRedirectUrl.searchParams.set(
      "message",
      `✅ Discord account ${username} linked successfully!`
    );
    return NextResponse.redirect(successRedirectUrl);
  } catch (error) {
    const errorInstance =
      error instanceof Error ? error : new Error(String(error));
    Logger.error("Discord OAuth callback failed", errorInstance, {
      endpoint: "GET /api/discord/auth",
    });

    const errorMessage = errorInstance.message;
    const errorRedirectUrl = new URL("/settings", request.url);
    errorRedirectUrl.searchParams.set("error", "discord_oauth_failed");
    errorRedirectUrl.searchParams.set(
      "message",
      `Failed to link Discord account: ${errorMessage}`
    );
    return NextResponse.redirect(errorRedirectUrl);
  }
}

import { NextRequest, NextResponse } from "next/server";

import { auth, isSignedIn } from "@/lib/auth";
import { discordAuth, DiscordAuthService } from "@/lib/discord-auth";
import { discordBot } from "@/lib/external-notifications/discord-bot";
import { prisma } from "@/lib/db/prisma";
import type { NotificationType } from "@prisma/client";

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
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_oauth_error");
      errorRedirectUrl.searchParams.set("message", error);
      return NextResponse.redirect(errorRedirectUrl);
    }

    if (!code) {
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_oauth_no_code");
      errorRedirectUrl.searchParams.set(
        "message",
        "No authorization code received"
      );
      return NextResponse.redirect(errorRedirectUrl);
    }

    // Exchange code for token
    console.log("🔄 Exchanging code for token...");
    const tokenResponse = await discordAuth.exchangeCodeForToken(code);
    console.log("✅ Token received");

    // Get Discord user info
    console.log("🔄 Getting Discord user info...");
    const discordUser = await discordAuth.getDiscordUser(
      tokenResponse.access_token
    );
    console.log("✅ Discord user:", discordUser.username, discordUser.id);

    // Check if this Discord account is already linked to another user
    console.log("🔄 Checking for existing links...");
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
    console.log("🔍 Existing link found:", !!existingLink);

    if (existingLink && existingLink.userId !== session.user.id) {
      console.log("❌ Discord account already linked to another user");
      const errorRedirectUrl = new URL("/settings", request.url);
      errorRedirectUrl.searchParams.set("error", "discord_already_linked");
      errorRedirectUrl.searchParams.set(
        "message",
        "This Discord account is already linked to another user"
      );
      return NextResponse.redirect(errorRedirectUrl);
    }

    // Link Discord account to user for all notification types
    console.log("🔄 Linking Discord account to user...");
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
    console.log(
      "📝 Generated username:",
      username,
      "displayName:",
      displayName
    );

    console.log("🔄 Starting database transaction...");
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
    console.log("✅ Database transaction completed");

    // Send welcome message
    console.log("🔄 Sending welcome message...");
    const welcomeSent = await discordBot.sendWelcomeMessage(
      discordUser.id,
      displayName
    );
    console.log("📨 Welcome message sent:", welcomeSent);

    // Revoke the access token (we don't need to store it since we only needed user info)
    console.log("🔄 Revoking access token...");
    await discordAuth.revokeToken(tokenResponse.access_token);
    console.log("✅ Access token revoked");

    // Redirect back to settings with success message
    console.log("🔄 Redirecting to settings with success message...");
    const successRedirectUrl = new URL("/settings", request.url);
    successRedirectUrl.searchParams.set("success", "discord_linked");
    successRedirectUrl.searchParams.set(
      "message",
      `✅ Discord account ${username} linked successfully!`
    );
    console.log("✅ Discord OAuth callback completed successfully");
    return NextResponse.redirect(successRedirectUrl);
  } catch (error) {
    console.error("❌ Error in Discord OAuth callback:", error);

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    console.error("❌ Error details:", errorMessage);

    const errorRedirectUrl = new URL("/settings", request.url);
    errorRedirectUrl.searchParams.set("error", "discord_oauth_failed");
    errorRedirectUrl.searchParams.set(
      "message",
      `Failed to link Discord account: ${errorMessage}`
    );
    return NextResponse.redirect(errorRedirectUrl);
  }
}

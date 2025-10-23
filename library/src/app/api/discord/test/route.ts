import { NextResponse } from "next/server";

import { discordBot } from "@/lib/external-notifications/discord-bot";
import { withErrorHandler } from "@/lib/api/error-handler";

// GET /api/discord/test - Check Discord bot configuration status
export const GET = withErrorHandler(async () => {
  const config = await discordBot.getConfigurationInfo();
  const isValid = await discordBot.testConfiguration();

  return NextResponse.json({
    configuration: config,
    isValid,
    message: isValid
      ? `✅ Discord bot is ready as ${config.botUser}`
      : `❌ Discord bot configuration invalid`,
  });
});

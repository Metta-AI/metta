import { fetchJson } from "@/lib/api/client";

/**
 * Discord connection status
 */
export interface DiscordStatus {
  isLinked: boolean;
  discordUsername?: string;
  discordId?: string;
}

/**
 * Discord bot configuration status
 */
export interface DiscordBotStatus {
  enabled: boolean;
  ready: boolean;
  botUser: string | null;
  username?: string;
  status?: string;
  error?: string;
}

/**
 * Notification preferences
 */
export interface NotificationPreferences {
  [key: string]: {
    emailEnabled: boolean;
    discordEnabled: boolean;
  };
}

/**
 * Get Discord connection status for current user
 */
export async function getDiscordStatus(): Promise<DiscordStatus> {
  return fetchJson<DiscordStatus>("/api/discord/link");
}

/**
 * Unlink Discord account
 */
export async function unlinkDiscord(): Promise<void> {
  await fetchJson("/api/discord/link", {
    method: "DELETE",
    skipJsonParse: true,
  });
}

/**
 * Get Discord bot configuration status
 */
export async function getDiscordBotStatus(): Promise<{
  configuration: DiscordBotStatus;
}> {
  return fetchJson("/api/discord/test");
}

/**
 * Get notification preferences for current user
 */
export async function getNotificationPreferences(): Promise<{
  preferences: NotificationPreferences;
}> {
  return fetchJson("/api/notification-preferences");
}

/**
 * Update notification preferences
 */
export async function updateNotificationPreferences(
  preferences: NotificationPreferences
): Promise<{ success: boolean }> {
  return fetchJson("/api/notification-preferences", {
    method: "POST",
    body: JSON.stringify({ preferences }),
  });
}

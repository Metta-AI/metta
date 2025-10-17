"use client";

import { useQuery } from "@tanstack/react-query";

interface DiscordStatusResponse {
  isLinked: boolean;
  discordUsername: string | null;
  discordUserId: string | null;
  discordLinkedAt: string | null;
  enabledNotificationTypes?: string[];
  message: string;
}

interface BotStatusResponse {
  configuration: {
    botEnabled?: boolean;
    clientId?: string | null;
    guildId?: string | null;
    enabled?: boolean;
    ready: boolean;
    botUser: string | null;
    username?: string;
    status?: string;
    error?: string;
  };
}

interface NotificationPreferencesResponse {
  preferences: Record<
    string,
    {
      emailEnabled: boolean;
      discordEnabled: boolean;
    }
  >;
}

export function useDiscordStatus() {
  return useQuery({
    queryKey: ["discord-status"],
    queryFn: async () => {
      const response = await fetch("/api/discord/link");
      if (!response.ok) {
        throw new Error("Failed to fetch Discord status");
      }
      return response.json() as Promise<DiscordStatusResponse>;
    },
  });
}

export function useBotStatus() {
  return useQuery({
    queryKey: ["bot-status"],
    queryFn: async () => {
      const response = await fetch("/api/discord/test");
      if (!response.ok) {
        throw new Error("Failed to fetch bot status");
      }
      return response.json() as Promise<BotStatusResponse>;
    },
  });
}

export function useNotificationPreferences() {
  return useQuery({
    queryKey: ["notification-preferences"],
    queryFn: async () => {
      const response = await fetch("/api/notification-preferences");
      if (!response.ok) {
        throw new Error("Failed to fetch notification preferences");
      }
      return response.json() as Promise<NotificationPreferencesResponse>;
    },
  });
}

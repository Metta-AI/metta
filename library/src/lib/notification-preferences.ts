/**
 * Notification Preferences Service
 *
 * Manages user notification preferences for different channels and types
 */

import { prisma } from "@/lib/db/prisma";
import type { NotificationType } from "@prisma/client";

export interface NotificationPreferences {
  [key: string]: {
    emailEnabled: boolean;
    discordEnabled: boolean;
  };
}

export interface UserNotificationPreferences {
  userId: string;
  preferences: NotificationPreferences;
}

/**
 * Get all notification preferences for a user
 */
export async function getUserNotificationPreferences(
  userId: string
): Promise<UserNotificationPreferences> {
  const preferences = await prisma.notificationPreference.findMany({
    where: { userId },
  });

  // Convert to object format
  const preferencesMap: NotificationPreferences = {};

  // Define all available notification types
  const notificationTypes: NotificationType[] = [
    "MENTION",
    "COMMENT",
    "REPLY",
    "LIKE",
    "FOLLOW",
    "PAPER_SUGGESTION",
    "SYSTEM",
  ];

  // Initialize all types with defaults, then override with user preferences
  for (const type of notificationTypes) {
    preferencesMap[type] = {
      emailEnabled: true, // Default to enabled
      discordEnabled: false, // Default to disabled
    };
  }

  // Apply user's actual preferences
  for (const pref of preferences) {
    preferencesMap[pref.type] = {
      emailEnabled: pref.emailEnabled,
      discordEnabled: pref.discordEnabled,
    };
  }

  return {
    userId,
    preferences: preferencesMap,
  };
}

/**
 * Get specific notification preference for a user and type
 */
export async function getNotificationPreference(
  userId: string,
  type: NotificationType
) {
  let preference = await prisma.notificationPreference.findUnique({
    where: {
      userId_type: { userId, type },
    },
  });

  // If no preference exists, create default one
  if (!preference) {
    preference = await prisma.notificationPreference.create({
      data: {
        userId,
        type,
        emailEnabled: true,
        discordEnabled: false,
      },
    });
  }

  return preference;
}

/**
 * Update notification preferences for a user
 */
export async function updateNotificationPreferences(
  userId: string,
  preferences: Partial<NotificationPreferences>
): Promise<void> {
  const updates = Object.entries(preferences).map(async ([type, settings]) => {
    return prisma.notificationPreference.upsert({
      where: {
        userId_type: {
          userId,
          type: type as NotificationType,
        },
      },
      create: {
        userId,
        type: type as NotificationType,
        emailEnabled: settings.emailEnabled ?? true,
        discordEnabled: settings.discordEnabled ?? false,
      },
      update: {
        emailEnabled: settings.emailEnabled,
        discordEnabled: settings.discordEnabled,
      },
    });
  });

  await Promise.all(updates);
}

/**
 * Update preference for a specific notification type
 */
export async function updateNotificationPreference(
  userId: string,
  type: NotificationType,
  settings: { emailEnabled?: boolean; discordEnabled?: boolean }
): Promise<void> {
  await prisma.notificationPreference.upsert({
    where: {
      userId_type: { userId, type },
    },
    create: {
      userId,
      type,
      emailEnabled: settings.emailEnabled ?? true,
      discordEnabled: settings.discordEnabled ?? false,
    },
    update: {
      emailEnabled: settings.emailEnabled,
      discordEnabled: settings.discordEnabled,
    },
  });
}

/**
 * Get enabled channels for a user and notification type
 */
export async function getEnabledChannels(
  userId: string,
  type: NotificationType
): Promise<("email" | "discord")[]> {
  const preference = await getNotificationPreference(userId, type);

  const channels: ("email" | "discord")[] = [];
  if (preference.emailEnabled) channels.push("email");

  // Only include Discord if enabled AND user has Discord account linked
  if (preference.discordEnabled && preference.discordUserId) {
    channels.push("discord");
  }

  return channels;
}

/**
 * Check if a specific channel is enabled for a user and notification type
 */
export async function isChannelEnabled(
  userId: string,
  type: NotificationType,
  channel: "email" | "discord"
): Promise<boolean> {
  const preference = await getNotificationPreference(userId, type);

  if (channel === "email") return preference.emailEnabled;
  if (channel === "discord") return preference.discordEnabled;

  return false;
}

/**
 * Disable all notifications for a user (useful for account deletion)
 */
export async function disableAllNotifications(userId: string): Promise<void> {
  await prisma.notificationPreference.updateMany({
    where: { userId },
    data: {
      emailEnabled: false,
      discordEnabled: false,
    },
  });
}

/**
 * Get notification delivery statistics for a user
 */
export async function getDeliveryStats(userId: string, days: number = 30) {
  const since = new Date();
  since.setDate(since.getDate() - days);

  const deliveries = await prisma.notificationDelivery.findMany({
    where: {
      notification: { userId },
      createdAt: { gte: since },
    },
    include: {
      notification: {
        select: { type: true },
      },
    },
  });

  // Group by channel and status
  const stats = {
    email: { sent: 0, failed: 0, pending: 0 },
    discord: { sent: 0, failed: 0, pending: 0 },
    byType: {} as Record<
      string,
      { sent: number; failed: number; pending: number }
    >,
  };

  for (const delivery of deliveries) {
    const channel = delivery.channel as "email" | "discord";
    const type = delivery.notification.type;

    // Initialize type stats if needed
    if (!stats.byType[type]) {
      stats.byType[type] = { sent: 0, failed: 0, pending: 0 };
    }

    if (delivery.status === "sent") {
      stats[channel].sent++;
      stats.byType[type].sent++;
    } else if (delivery.status === "failed") {
      stats[channel].failed++;
      stats.byType[type].failed++;
    } else {
      stats[channel].pending++;
      stats.byType[type].pending++;
    }
  }

  return stats;
}

/**
 * Get failed deliveries that need attention
 */
export async function getFailedDeliveries(
  userId?: string,
  limit: number = 100
) {
  return prisma.notificationDelivery.findMany({
    where: {
      status: "failed",
      ...(userId && {
        notification: { userId },
      }),
    },
    include: {
      notification: {
        select: {
          id: true,
          type: true,
          title: true,
          userId: true,
          createdAt: true,
        },
      },
    },
    orderBy: { createdAt: "desc" },
    take: limit,
  });
}

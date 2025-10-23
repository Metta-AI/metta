import { prisma } from "@/lib/db/prisma";
import { ResolvedMention } from "./mention-resolution";
import { JobQueueService, type NotificationData } from "./job-queue";
import {
  getEnabledChannels,
  type NotificationPreferenceSettings,
} from "./notification-preferences";
import { Logger } from "./logging/logger";

export type NotificationType =
  | "MENTION"
  | "COMMENT"
  | "REPLY"
  | "LIKE"
  | "FOLLOW"
  | "PAPER_SUGGESTION"
  | "SYSTEM";

interface CreateNotificationParams {
  userId: string;
  type: NotificationType;
  title: string;
  message?: string;
  actionUrl?: string;
  actorId?: string;
  postId?: string;
  commentId?: string;
  mentionText?: string;
}

/**
 * Create a single notification record with full data fetching
 */
export async function createNotification(params: CreateNotificationParams) {
  // Create notification and immediately fetch with all relations
  const notification = await prisma.notification.create({
    data: {
      userId: params.userId,
      type: params.type,
      title: params.title,
      message: params.message,
      actionUrl: params.actionUrl,
      actorId: params.actorId,
      postId: params.postId,
      commentId: params.commentId,
      mentionText: params.mentionText,
    },
    include: {
      user: {
        select: { id: true, name: true, email: true },
      },
      actor: {
        select: { id: true, name: true, email: true },
      },
      post: {
        select: { id: true, title: true },
      },
      comment: {
        select: {
          id: true,
          content: true,
          post: { select: { id: true, title: true } },
        },
      },
    },
  });

  // Queue external notifications (email, Discord) if enabled
  await queueExternalNotifications(notification);

  return notification;
}

/**
 * Queue external notifications with full notification data (avoid re-fetching)
 */
async function queueExternalNotifications(
  notification: NotificationData
): Promise<void> {
  try {
    // Get enabled channels for this user and notification type
    const enabledChannels = await getEnabledChannels(
      notification.userId,
      notification.type
    );

    if (enabledChannels.length === 0) {
      Logger.debug("No enabled channels for notification", {
        notificationId: notification.id,
        userId: notification.userId,
      });
      return;
    }

    // Get user preferences once
    const preferences: NotificationPreferenceSettings = {
      emailEnabled: enabledChannels.includes("email"),
      discordEnabled: enabledChannels.includes("discord"),
    };

    // Determine priority based on notification type
    const priority =
      notification.type === "SYSTEM"
        ? 10
        : notification.type === "MENTION"
          ? 5
          : 0;

    // Queue with full notification data - no DB queries needed in worker
    await JobQueueService.queueExternalNotification(
      notification,
      enabledChannels,
      preferences,
      priority
    );

    Logger.debug("Queued external notifications", {
      notificationId: notification.id,
      channels: enabledChannels,
    });
  } catch (error) {
    Logger.error(
      "Failed to queue external notifications",
      error instanceof Error ? error : new Error(String(error)),
      { notificationId: notification.id }
    );
    // Don't throw - we don't want external notification failures to break the main flow
  }
}

/**
 * Create multiple notifications in batch with full data fetching
 */
export async function createNotifications(
  notifications: CreateNotificationParams[]
) {
  if (notifications.length === 0) return [];

  // Create notifications and fetch with relations
  const result = await prisma.$transaction(async (tx) => {
    const createdNotifications = [];

    for (const notificationData of notifications) {
      const notification = await tx.notification.create({
        data: notificationData,
        include: {
          user: {
            select: { id: true, name: true, email: true },
          },
          actor: {
            select: { id: true, name: true, email: true },
          },
          post: {
            select: { id: true, title: true },
          },
          comment: {
            select: {
              id: true,
              content: true,
              post: { select: { id: true, title: true } },
            },
          },
        },
      });
      createdNotifications.push(notification);
    }

    return createdNotifications;
  });

  // Queue external notifications for each created notification (parallel)
  const queuePromises = result.map((notification) =>
    queueExternalNotifications(notification)
  );

  // Await in parallel with proper error handling
  await Promise.allSettled(queuePromises).then((results) => {
    const failures = results.filter((r) => r.status === "rejected");
    if (failures.length > 0) {
      Logger.warn("Some external notification queuing failed", {
        failureCount: failures.length,
        totalCount: results.length,
      });
    }
  });

  return result;
}

/**
 * Create mention notifications from resolved mentions
 */
export async function createMentionNotifications(
  resolvedMentions: ResolvedMention[],
  actorId: string,
  actorName: string,
  contentType: "post" | "comment",
  contentId: string,
  actionUrl: string
) {
  const notifications: CreateNotificationParams[] = [];

  for (const mention of resolvedMentions) {
    const baseTitle =
      contentType === "post"
        ? `${actorName} mentioned you in a post`
        : `${actorName} mentioned you in a comment`;

    if (mention.type === "user") {
      // Individual user mention
      for (const userId of mention.userIds) {
        notifications.push({
          userId,
          type: "MENTION",
          title: baseTitle,
          message: `You were mentioned: "${mention.originalMention}"`,
          actionUrl,
          actorId,
          postId: contentType === "post" ? contentId : undefined,
          commentId: contentType === "comment" ? contentId : undefined,
          mentionText: mention.originalMention,
        });
      }
    } else if (mention.type === "group") {
      // Group mention
      const groupInfo = mention.institutionName
        ? `${mention.groupName} (${mention.institutionName})`
        : mention.groupName;

      const groupTitle =
        contentType === "post"
          ? `${actorName} mentioned ${groupInfo} in a post`
          : `${actorName} mentioned ${groupInfo} in a comment`;

      for (const userId of mention.userIds) {
        notifications.push({
          userId,
          type: "MENTION",
          title: groupTitle,
          message: `Your group was mentioned: "${mention.originalMention}"`,
          actionUrl,
          actorId,
          postId: contentType === "post" ? contentId : undefined,
          commentId: contentType === "comment" ? contentId : undefined,
          mentionText: mention.originalMention,
        });
      }
    }
  }

  if (notifications.length > 0) {
    await createNotifications(notifications);
    Logger.info("Created mention notifications", {
      count: notifications.length,
    });
  }

  return notifications;
}

/**
 * Mark notifications as read
 */
export async function markNotificationsRead(notificationIds: string[]) {
  return await prisma.notification.updateMany({
    where: {
      id: { in: notificationIds },
    },
    data: {
      isRead: true,
      updatedAt: new Date(),
    },
  });
}

/**
 * Mark all notifications for a user as read
 */
export async function markAllNotificationsRead(userId: string) {
  return await prisma.notification.updateMany({
    where: {
      userId,
      isRead: false,
    },
    data: {
      isRead: true,
      updatedAt: new Date(),
    },
  });
}

/**
 * Get notification count for a user
 */
export async function getNotificationCounts(userId: string) {
  const [total, unread] = await Promise.all([
    prisma.notification.count({
      where: { userId },
    }),
    prisma.notification.count({
      where: { userId, isRead: false },
    }),
  ]);

  return { total, unread };
}

/**
 * Get notifications for a user with pagination
 */
export async function getUserNotifications(
  userId: string,
  options: {
    limit?: number;
    offset?: number;
    includeRead?: boolean;
  } = {}
) {
  const { limit = 20, offset = 0, includeRead = true } = options;

  return await prisma.notification.findMany({
    where: {
      userId,
      ...(includeRead ? {} : { isRead: false }),
    },
    include: {
      actor: {
        select: {
          id: true,
          name: true,
          email: true,
        },
      },
      post: {
        select: {
          id: true,
          title: true,
        },
      },
      comment: {
        select: {
          id: true,
          content: true,
          post: {
            select: {
              id: true,
              title: true,
            },
          },
        },
      },
    },
    orderBy: {
      createdAt: "desc",
    },
    skip: offset,
    take: limit,
  });
}

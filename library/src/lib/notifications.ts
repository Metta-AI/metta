import { prisma } from "@/lib/db/prisma";
import { ResolvedMention } from "./mention-resolution";
import { JobQueueService } from "./job-queue";
import { getEnabledChannels } from "./notification-preferences";

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
 * Create a single notification record
 */
export async function createNotification(params: CreateNotificationParams) {
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
  });

  // Queue external notifications (email, Discord) if enabled
  await queueExternalNotifications(
    notification.id,
    notification.userId,
    notification.type
  );

  return notification;
}

/**
 * Queue external notifications for a notification
 */
async function queueExternalNotifications(
  notificationId: string,
  userId: string,
  type: NotificationType
): Promise<void> {
  try {
    // Get enabled channels for this user and notification type
    const enabledChannels = await getEnabledChannels(userId, type);

    if (enabledChannels.length > 0) {
      // Determine priority based on notification type
      const priority = type === "SYSTEM" ? 10 : type === "MENTION" ? 5 : 0;

      await JobQueueService.queueExternalNotification(
        notificationId,
        enabledChannels,
        userId,
        priority
      );

      console.log(
        `📤 Queued external notifications for ${notificationId}: ${enabledChannels.join(", ")}`
      );
    }
  } catch (error) {
    console.error(
      `❌ Failed to queue external notifications for ${notificationId}:`,
      error
    );
    // Don't throw - we don't want external notification failures to break the main flow
  }
}

/**
 * Create multiple notifications in batch
 */
export async function createNotifications(
  notifications: CreateNotificationParams[]
) {
  if (notifications.length === 0) return [];

  // Create notifications and get the IDs back
  const result = await prisma.$transaction(async (tx) => {
    const createdNotifications = [];

    for (const notificationData of notifications) {
      const notification = await tx.notification.create({
        data: notificationData,
      });
      createdNotifications.push(notification);
    }

    return createdNotifications;
  });

  // Queue external notifications for each created notification
  const queuePromises = result.map((notification) =>
    queueExternalNotifications(
      notification.id,
      notification.userId,
      notification.type
    )
  );

  // Don't await these - let them run in background
  Promise.allSettled(queuePromises).catch((error) => {
    console.error("❌ Some external notification queuing failed:", error);
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
    console.log(`📧 Created ${notifications.length} mention notifications`);
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

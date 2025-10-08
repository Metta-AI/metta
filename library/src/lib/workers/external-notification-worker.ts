/**
 * External Notification Worker
 *
 * Processes jobs from the external-notifications queue using BullMQ
 * Handles email and Discord notification delivery
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { emailService } from "../external-notifications/email";
import { discordBot } from "../external-notifications/discord-bot";
import { prisma } from "@/lib/db/prisma";
import { Logger } from "../logging/logger";

export class ExternalNotificationWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      "external-notifications",
      async (
        job: Job<
          | BackgroundJobs["send-external-notification"]
          | BackgroundJobs["retry-failed-notification"]
        >
      ) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: 3, // Process up to 3 notifications concurrently
        limiter: {
          max: 20, // Max 20 notifications per
          duration: 60000, // Per minute (rate limiting for external services)
        },
      }
    );

    this.setupEventHandlers();
  }

  private async processJob(
    job: Job<
      | BackgroundJobs["send-external-notification"]
      | BackgroundJobs["retry-failed-notification"]
    >
  ): Promise<void> {
    if (job.name === "send-external-notification") {
      await this.processSendNotification(
        job as Job<BackgroundJobs["send-external-notification"]>
      );
    } else if (job.name === "retry-failed-notification") {
      await this.processRetryNotification(
        job as Job<BackgroundJobs["retry-failed-notification"]>
      );
    }
  }

  private async processSendNotification(
    job: Job<BackgroundJobs["send-external-notification"]>
  ): Promise<void> {
    const { notificationId, channels, userId } = job.data;

    Logger.info("External Notification Worker: Processing notification", {
      notificationId,
      userId,
      channels,
    });

    try {
      // Fetch notification with all required relations
      const notification = await prisma.notification.findUnique({
        where: { id: notificationId },
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

      if (!notification) {
        Logger.error("Notification not found", new Error("Not found"), {
          notificationId,
        });
        return;
      }

      // Get user notification preferences
      const preferences = await this.getUserPreferences(
        userId,
        notification.type
      );

      // Filter channels based on user preferences
      const enabledChannels = channels.filter((channel) => {
        if (channel === "email") return preferences.emailEnabled;
        if (channel === "discord") return preferences.discordEnabled;
        return false;
      });

      if (enabledChannels.length === 0) {
        Logger.debug("No enabled channels for notification", {
          userId,
          notificationId,
        });
        return;
      }

      // Process each enabled channel
      const results = await Promise.allSettled(
        enabledChannels.map(async (channel) => {
          if (channel === "email") {
            return await this.sendEmailNotification(notification);
          } else if (channel === "discord") {
            return await this.sendDiscordNotification(notification);
          }
          return false;
        })
      );

      // Log results
      results.forEach((result, index) => {
        const channel = enabledChannels[index];
        if (result.status === "fulfilled") {
          Logger.info("External notification sent successfully", {
            channel,
            notificationId,
          });
        } else {
          Logger.error(
            "External notification failed",
            result.reason instanceof Error
              ? result.reason
              : new Error(String(result.reason)),
            { channel, notificationId }
          );
        }
      });
    } catch (error) {
      Logger.error(
        "External Notification Worker: Failed processing notification",
        error instanceof Error ? error : new Error(String(error)),
        { notificationId }
      );
      throw error; // BullMQ will handle retries
    }
  }

  private async processRetryNotification(
    job: Job<BackgroundJobs["retry-failed-notification"]>
  ): Promise<void> {
    const { deliveryId } = job.data;

    Logger.info("External Notification Worker: Retrying delivery", {
      deliveryId,
    });

    try {
      // Fetch delivery record with notification
      const delivery = await prisma.notificationDelivery.findUnique({
        where: { id: deliveryId },
        include: {
          notification: {
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
          },
        },
      });

      if (!delivery) {
        Logger.error("Delivery not found for retry", new Error("Not found"), {
          deliveryId,
        });
        return;
      }

      // Retry based on channel
      let success = false;
      if (delivery.channel === "email") {
        success = await emailService.sendNotification(
          delivery.notification,
          deliveryId
        );
      } else if (delivery.channel === "discord") {
        // Get Discord user ID for this notification
        const preferences = await this.getUserPreferences(
          delivery.notification.userId,
          delivery.notification.type
        );

        if (preferences.discordUserId) {
          success = await discordBot.sendNotification(
            delivery.notification,
            preferences.discordUserId,
            deliveryId
          );
        }
      }

      if (success) {
        Logger.info("External Notification Worker: Retry successful", {
          deliveryId,
        });
      } else {
        Logger.warn("External Notification Worker: Retry failed", {
          deliveryId,
        });
      }
    } catch (error) {
      Logger.error(
        "External Notification Worker: Failed retry for delivery",
        error instanceof Error ? error : new Error(String(error)),
        { deliveryId }
      );
      throw error;
    }
  }

  private async sendEmailNotification(notification: any): Promise<boolean> {
    if (!notification.user.email) {
      Logger.warn("No email address for user", {
        userId: notification.userId,
      });
      return false;
    }

    return await emailService.sendNotification(notification);
  }

  private async sendDiscordNotification(notification: any): Promise<boolean> {
    // Get user's Discord preferences to find their Discord user ID
    const preferences = await this.getUserPreferences(
      notification.userId,
      notification.type
    );

    if (!preferences.discordUserId) {
      Logger.warn(
        `🤖 No Discord account linked for user ${notification.userId}`
      );
      return false;
    }

    return await discordBot.sendNotification(
      notification,
      preferences.discordUserId
    );
  }

  private async getUserPreferences(userId: string, notificationType: string) {
    // Try to find user's specific preference for this notification type
    let preference = await prisma.notificationPreference.findUnique({
      where: {
        userId_type: {
          userId,
          type: notificationType as any,
        },
      },
    });

    // If no specific preference exists, create default one
    if (!preference) {
      preference = await prisma.notificationPreference.create({
        data: {
          userId,
          type: notificationType as any,
          emailEnabled: true, // Default to enabled
          discordEnabled: false, // Default to disabled
        },
      });
    }

    return preference;
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      Logger.info(
        `✅ External notification job ${job.id} completed successfully`
      );
    });

    this.worker.on("failed", (job, err) => {
      Logger.error(
        `❌ External notification job ${job?.id} failed:`,
        err.message
      );
    });

    this.worker.on("error", (err) => {
      Logger.error("🚨 External notification worker error:", err);
    });
  }

  async shutdown(): Promise<void> {
    Logger.info("🛑 Shutting down external notification worker...");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startExternalNotificationWorker(): Promise<ExternalNotificationWorker> {
  const worker = new ExternalNotificationWorker();
  Logger.info("🚀 External notification worker started");
  return worker;
}

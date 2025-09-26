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

    console.log(
      `📧 [External Notification Worker] Processing notification ${notificationId} for user ${userId}`
    );
    console.log(`   Channels: ${channels.join(", ")}`);

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
        console.error(`❌ Notification ${notificationId} not found`);
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
        console.log(
          `📧 No enabled channels for user ${userId}, notification ${notificationId}`
        );
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
          console.log(
            `✅ [External Notification Worker] ${channel} notification sent for ${notificationId}`
          );
        } else {
          console.error(
            `❌ [External Notification Worker] ${channel} notification failed for ${notificationId}:`,
            result.reason
          );
        }
      });
    } catch (error) {
      console.error(
        `❌ [External Notification Worker] Failed processing notification ${notificationId}:`,
        error
      );
      throw error; // BullMQ will handle retries
    }
  }

  private async processRetryNotification(
    job: Job<BackgroundJobs["retry-failed-notification"]>
  ): Promise<void> {
    const { deliveryId } = job.data;

    console.log(
      `🔄 [External Notification Worker] Retrying delivery ${deliveryId}`
    );

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
        console.error(`❌ Delivery ${deliveryId} not found`);
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
        console.log(
          `✅ [External Notification Worker] Retry successful for delivery ${deliveryId}`
        );
      } else {
        console.error(
          `❌ [External Notification Worker] Retry failed for delivery ${deliveryId}`
        );
      }
    } catch (error) {
      console.error(
        `❌ [External Notification Worker] Failed retry for delivery ${deliveryId}:`,
        error
      );
      throw error;
    }
  }

  private async sendEmailNotification(notification: any): Promise<boolean> {
    if (!notification.user.email) {
      console.warn(`📧 No email address for user ${notification.userId}`);
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
      console.warn(
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
      console.log(
        `✅ External notification job ${job.id} completed successfully`
      );
    });

    this.worker.on("failed", (job, err) => {
      console.error(
        `❌ External notification job ${job?.id} failed:`,
        err.message
      );
    });

    this.worker.on("error", (err) => {
      console.error("🚨 External notification worker error:", err);
    });
  }

  async shutdown(): Promise<void> {
    console.log("🛑 Shutting down external notification worker...");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startExternalNotificationWorker(): Promise<ExternalNotificationWorker> {
  const worker = new ExternalNotificationWorker();
  console.log("🚀 External notification worker started");
  return worker;
}

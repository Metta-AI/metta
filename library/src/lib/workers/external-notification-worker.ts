/**
 * External Notification Worker
 *
 * Processes jobs from the external-notifications queue using BullMQ
 * Handles email and Discord notification delivery
 */

import { Job } from "bullmq";
import { BackgroundJobs } from "../job-queue";
import { emailService } from "../external-notifications/email";
import { discordBot } from "../external-notifications/discord-bot";
import { prisma } from "@/lib/db/prisma";
import { Logger } from "../logging/logger";
import { BaseWorker } from "./base-worker";

export class ExternalNotificationWorker extends BaseWorker<
  | BackgroundJobs["send-external-notification"]
  | BackgroundJobs["retry-failed-notification"]
> {
  constructor() {
    super({
      queueName: "external-notifications",
      concurrency: 3, // Process up to 3 notifications concurrently
      maxJobsPerMinute: 20, // Rate limiting for external services
    });
  }

  protected async processJob(
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
    const { notification, channels, preferences } = job.data;

    Logger.info("External Notification Worker: Processing notification", {
      notificationId: notification.id,
      userId: notification.userId,
      channels,
    });

    try {
      // No DB queries needed - all data is in the job payload!

      // Filter channels based on preferences (already fetched)
      const enabledChannels = channels.filter((channel) => {
        if (channel === "email") return preferences.emailEnabled;
        if (channel === "discord") return preferences.discordEnabled;
        return false;
      });

      if (enabledChannels.length === 0) {
        Logger.debug("No enabled channels for notification", {
          userId: notification.userId,
          notificationId: notification.id,
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
            notificationId: notification.id,
          });
        } else {
          Logger.error(
            "External notification failed",
            result.reason instanceof Error
              ? result.reason
              : new Error(String(result.reason)),
            { channel, notificationId: notification.id }
          );
        }
      });
    } catch (error) {
      Logger.error(
        "External Notification Worker: Failed processing notification",
        error instanceof Error ? error : new Error(String(error)),
        { notificationId: notification.id }
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
        // Get Discord user ID for retry
        const discordPreference = await prisma.notificationPreference.findFirst(
          {
            where: {
              userId: delivery.notification.userId,
              discordUserId: { not: null },
            },
            select: { discordUserId: true },
          }
        );

        if (discordPreference?.discordUserId) {
          success = await discordBot.sendNotification(
            delivery.notification,
            discordPreference.discordUserId,
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
    if (!notification.user?.email) {
      Logger.warn("No email address for user", {
        userId: notification.userId,
      });
      return false;
    }

    return await emailService.sendNotification(notification);
  }

  private async sendDiscordNotification(notification: any): Promise<boolean> {
    // Get user's Discord user ID from notification preferences
    const discordPreference = await prisma.notificationPreference.findFirst({
      where: {
        userId: notification.userId,
        discordUserId: { not: null },
      },
      select: { discordUserId: true },
    });

    if (!discordPreference?.discordUserId) {
      Logger.warn(
        `🤖 No Discord account linked for user ${notification.userId}`
      );
      return false;
    }

    return await discordBot.sendNotification(
      notification,
      discordPreference.discordUserId
    );
  }

  protected setupEventHandlers(): void {
    // Override base event handlers with custom formatting
    this.worker.on("completed", (job) => {
      Logger.info("External notification job completed", { jobId: job.id });
    });

    this.worker.on("failed", (job, err) => {
      Logger.error("External notification job failed", err, {
        jobId: job?.id,
      });
    });

    this.worker.on("error", (err) => {
      Logger.error("External notification worker error", err);
    });
  }
}

// Export for standalone usage
export async function startExternalNotificationWorker(): Promise<ExternalNotificationWorker> {
  const worker = new ExternalNotificationWorker();
  Logger.info("🚀 External notification worker started");
  return worker;
}

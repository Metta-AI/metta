/**
 * Job Queue Configuration using BullMQ + Redis
 *
 * This replaces the previous setImmediate() approach with proper job queuing
 * for background processing of institutions, authors, and LLM abstracts.
 */

import { Queue, Worker, Job } from "bullmq";
import { Redis } from "ioredis";
import { config } from "./config";
import { Logger } from "./logging/logger";

// Redis connection configuration
const redisConfig = {
  host: config.redis.host,
  port: config.redis.port,
  ...(config.redis.password && { password: config.redis.password }),
  ...(config.redis.tls && { tls: {} }),
  connectTimeout: 10000,
  lazyConnect: false,
};

// Job type definitions
export interface BackgroundJobs {
  "extract-institutions": {
    paperId: string;
    arxivUrl: string;
  };
  "extract-authors": {
    paperId: string;
    arxivUrl: string;
  };
  "generate-llm-abstract": {
    paperId: string;
  };
  "auto-tag-paper": {
    paperId: string;
  };
  "send-external-notification": {
    notificationId: string;
    channels: ("email" | "discord")[];
    userId: string;
  };
  "retry-failed-notification": {
    deliveryId: string;
  };
}

// Job queue instances
export const institutionQueue = new Queue("institution-extraction", {
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: 100, // Keep last 100 successful jobs
    removeOnFail: 50, // Keep last 50 failed jobs
    attempts: 3, // Retry up to 3 times
    backoff: {
      type: "exponential",
      delay: 2000, // Start with 2s delay, then 4s, 8s
    },
  },
});

export const authorQueue = new Queue("author-extraction", {
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: 100,
    removeOnFail: 50,
    attempts: 3,
    backoff: {
      type: "exponential",
      delay: 2000,
    },
  },
});

export const llmQueue = new Queue("llm-processing", {
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: 50,
    removeOnFail: 25,
    attempts: 2, // LLM jobs are expensive, fewer retries
    backoff: {
      type: "exponential",
      delay: 5000, // Longer delays for LLM failures
    },
  },
});

export const taggingQueue = new Queue("auto-tagging", {
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: 100,
    removeOnFail: 50,
    attempts: 3,
    backoff: {
      type: "exponential",
      delay: 1000,
    },
  },
});

export const externalNotificationQueue = new Queue("external-notifications", {
  connection: redisConfig,
  defaultJobOptions: {
    removeOnComplete: 200, // Keep more notification jobs for audit
    removeOnFail: 100,
    attempts: 3,
    backoff: {
      type: "exponential",
      delay: 2000, // Start with 2s delay for failed notifications
    },
  },
});

/**
 * Queue background jobs for processing
 */
export class JobQueueService {
  /**
   * Queue institution extraction for a paper
   */
  static async queueInstitutionExtraction(
    paperId: string,
    arxivUrl: string
  ): Promise<void> {
    Logger.info(`📤 Queuing institution extraction for paper ${paperId}`);

    await institutionQueue.add(
      "extract-institutions",
      { paperId, arxivUrl },
      {
        // Rate limiting: process with 3 second delay
        delay: 3000,
      }
    );
  }

  /**
   * Queue author extraction for a paper
   */
  static async queueAuthorExtraction(
    paperId: string,
    arxivUrl: string
  ): Promise<void> {
    Logger.info(`📤 Queuing author extraction for paper ${paperId}`);

    await authorQueue.add(
      "extract-authors",
      { paperId, arxivUrl },
      {
        // Rate limiting: process with 2 second delay
        delay: 2000,
      }
    );
  }

  /**
   * Queue LLM abstract generation for a paper
   */
  static async queueLLMAbstractGeneration(paperId: string): Promise<void> {
    Logger.info(`📤 Queuing LLM abstract generation for paper ${paperId}`);

    await llmQueue.add(
      "generate-llm-abstract",
      { paperId },
      {
        // LLM jobs can run immediately, they're already expensive
        priority: 10, // Higher priority than other jobs
      }
    );
  }

  /**
   * Queue auto-tagging for a paper
   */
  static async queueAutoTagging(paperId: string): Promise<void> {
    Logger.info(`📤 Queuing auto-tagging for paper ${paperId}`);

    await taggingQueue.add(
      "auto-tag-paper",
      { paperId },
      {
        delay: 1000, // Small delay
      }
    );
  }

  /**
   * Queue external notification sending
   */
  static async queueExternalNotification(
    notificationId: string,
    channels: ("email" | "discord")[],
    userId: string,
    priority: number = 0
  ): Promise<void> {
    Logger.info(
      `📤 Queuing external notifications for ${notificationId}: ${channels.join(", ")}`
    );

    await externalNotificationQueue.add(
      "send-external-notification",
      { notificationId, channels, userId },
      {
        priority, // Higher priority notifications go first
        delay: 500, // Small delay to allow database to settle
      }
    );
  }

  /**
   * Queue retry for a failed notification delivery
   */
  static async queueNotificationRetry(deliveryId: string): Promise<void> {
    Logger.info(`📤 Queuing notification retry for delivery ${deliveryId}`);

    await externalNotificationQueue.add(
      "retry-failed-notification",
      { deliveryId },
      {
        priority: 5, // Higher priority for retries
        delay: 5000, // Wait 5 seconds before retry
      }
    );
  }

  /**
   * Get queue statistics for monitoring
   */
  static async getQueueStats() {
    const [
      institutionStats,
      authorStats,
      llmStats,
      taggingStats,
      notificationStats,
    ] = await Promise.all([
      institutionQueue.getJobCounts(),
      authorQueue.getJobCounts(),
      llmQueue.getJobCounts(),
      taggingQueue.getJobCounts(),
      externalNotificationQueue.getJobCounts(),
    ]);

    return {
      institution: institutionStats,
      author: authorStats,
      llm: llmStats,
      tagging: taggingStats,
      notifications: notificationStats,
    };
  }

  /**
   * Graceful shutdown - close all queues
   */
  static async shutdown(): Promise<void> {
    Logger.info("🛑 Shutting down job queues...");
    await Promise.all([
      institutionQueue.close(),
      authorQueue.close(),
      llmQueue.close(),
      taggingQueue.close(),
      externalNotificationQueue.close(),
    ]);
  }
}

// Export queue instances for worker creation
export { redisConfig };

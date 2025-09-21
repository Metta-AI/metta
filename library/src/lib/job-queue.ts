/**
 * Job Queue Configuration using BullMQ + Redis
 *
 * This replaces the previous setImmediate() approach with proper job queuing
 * for background processing of institutions, authors, and LLM abstracts.
 */

import { Queue, Worker, Job } from "bullmq";
import { Redis } from "ioredis";

// Redis connection configuration
const redisConfig = {
  host: process.env.REDIS_HOST || "localhost",
  port: parseInt(process.env.REDIS_PORT || "6379"),
  // Add auth if needed in production
  ...(process.env.REDIS_PASSWORD && { password: process.env.REDIS_PASSWORD }),
  // Add TLS if needed for ElastiCache encryption in transit
  ...(process.env.REDIS_TLS === "true" && { tls: {} }),
  // Connection timeout to prevent hanging
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
    console.log(`📤 Queuing institution extraction for paper ${paperId}`);

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
    console.log(`📤 Queuing author extraction for paper ${paperId}`);

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
    console.log(`📤 Queuing LLM abstract generation for paper ${paperId}`);

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
    console.log(`📤 Queuing auto-tagging for paper ${paperId}`);

    await taggingQueue.add(
      "auto-tag-paper",
      { paperId },
      {
        delay: 1000, // Small delay
      }
    );
  }

  /**
   * Get queue statistics for monitoring
   */
  static async getQueueStats() {
    const [institutionStats, authorStats, llmStats, taggingStats] =
      await Promise.all([
        institutionQueue.getJobCounts(),
        authorQueue.getJobCounts(),
        llmQueue.getJobCounts(),
        taggingQueue.getJobCounts(),
      ]);

    return {
      institution: institutionStats,
      author: authorStats,
      llm: llmStats,
      tagging: taggingStats,
    };
  }

  /**
   * Graceful shutdown - close all queues
   */
  static async shutdown(): Promise<void> {
    console.log("🛑 Shutting down job queues...");
    await Promise.all([
      institutionQueue.close(),
      authorQueue.close(),
      llmQueue.close(),
      taggingQueue.close(),
    ]);
  }
}

// Export queue instances for worker creation
export { redisConfig };

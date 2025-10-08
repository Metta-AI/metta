/**
 * Auto-Tagging Worker
 *
 * Processes jobs from the auto-tagging queue using BullMQ
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { AutoTaggingService } from "../auto-tagging-service";
import { Logger } from "../logging/logger";

export class TaggingWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      "auto-tagging",
      async (job: Job<BackgroundJobs["auto-tag-paper"]>) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: 3, // Can process multiple tagging jobs concurrently
        limiter: {
          max: 20, // Max 20 jobs
          duration: 60000, // Per minute
        },
      }
    );

    this.setupEventHandlers();
  }

  private async processJob(
    job: Job<BackgroundJobs["auto-tag-paper"]>
  ): Promise<void> {
    const { paperId } = job.data;

    Logger.info("Tagging Worker: Auto-tagging paper", { paperId });

    try {
      const tags = await AutoTaggingService.autoTagPaper(paperId);
      Logger.info("Tagging Worker: Tagged paper", {
        paperId,
        tagCount: tags?.length || 0,
      });
    } catch (error) {
      Logger.error(
        "Tagging Worker: Failed to tag paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId }
      );
      throw error; // BullMQ will handle retries
    }
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      Logger.debug("Tagging job completed", { jobId: job.id });
    });

    this.worker.on("failed", (job, err) => {
      Logger.error("Tagging job failed", err, { jobId: job?.id });
    });

    this.worker.on("error", (err) => {
      Logger.error("Tagging worker error", err);
    });
  }

  async shutdown(): Promise<void> {
    Logger.info("Shutting down tagging worker");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startTaggingWorker(): Promise<TaggingWorker> {
  const worker = new TaggingWorker();
  Logger.info("Tagging worker started");
  return worker;
}

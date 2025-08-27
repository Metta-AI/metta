/**
 * Auto-Tagging Worker
 *
 * Processes jobs from the auto-tagging queue using BullMQ
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { AutoTaggingService } from "../auto-tagging-service";

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

    console.log(`🏷️ [Tagging Worker] Auto-tagging paper ${paperId}`);

    try {
      const tags = await AutoTaggingService.autoTagPaper(paperId);
      console.log(
        `✅ [Tagging Worker] Tagged paper ${paperId} with ${tags?.length || 0} tags`
      );
    } catch (error) {
      console.error(
        `❌ [Tagging Worker] Failed to tag paper ${paperId}:`,
        error
      );
      throw error; // BullMQ will handle retries
    }
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      console.log(`✅ Tagging job ${job.id} completed successfully`);
    });

    this.worker.on("failed", (job, err) => {
      console.error(`❌ Tagging job ${job?.id} failed:`, err.message);
    });

    this.worker.on("error", (err) => {
      console.error("🚨 Tagging worker error:", err);
    });
  }

  async shutdown(): Promise<void> {
    console.log("🛑 Shutting down tagging worker...");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startTaggingWorker(): Promise<TaggingWorker> {
  const worker = new TaggingWorker();
  console.log("🚀 Tagging worker started");
  return worker;
}

/**
 * Auto-Tagging Worker
 *
 * Processes jobs from the auto-tagging queue using BullMQ
 */

import { Job } from "bullmq";
import { BackgroundJobs } from "../job-queue";
import { AutoTaggingService } from "../auto-tagging-service";
import { Logger } from "../logging/logger";
import { BaseWorker } from "./base-worker";

export class TaggingWorker extends BaseWorker<
  BackgroundJobs["auto-tag-paper"]
> {
  constructor() {
    super({
      queueName: "auto-tagging",
      concurrency: 3, // Can process multiple tagging jobs concurrently
      maxJobsPerMinute: 20,
    });
  }

  protected async processJob(
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
}

// Export for standalone usage
export async function startTaggingWorker(): Promise<TaggingWorker> {
  const worker = new TaggingWorker();
  Logger.info("Tagging worker started");
  return worker;
}

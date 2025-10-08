/**
 * Institution Extraction Worker
 *
 * Processes jobs from the institution-extraction queue using BullMQ
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { processArxivInstitutionsAsync } from "../arxiv-auto-import";
import { Logger } from "../logging/logger";

export class InstitutionWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      "institution-extraction",
      async (job: Job<BackgroundJobs["extract-institutions"]>) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: 2, // Process 2 jobs concurrently
        limiter: {
          max: 10, // Max 10 jobs
          duration: 60000, // Per minute (rate limiting for external APIs)
        },
      }
    );

    this.setupEventHandlers();
  }

  private async processJob(
    job: Job<BackgroundJobs["extract-institutions"]>
  ): Promise<void> {
    const { paperId, arxivUrl } = job.data;

    Logger.info("Institution Worker: Processing paper", { paperId, arxivUrl });

    try {
      await processArxivInstitutionsAsync(paperId, arxivUrl);
      Logger.info("Institution Worker: Completed paper", { paperId });
    } catch (error) {
      Logger.error(
        "Institution Worker: Failed to process paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId, arxivUrl }
      );
      throw error; // BullMQ will handle retries
    }
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      Logger.debug("Institution job completed", { jobId: job.id });
    });

    this.worker.on("failed", (job, err) => {
      Logger.error("Institution job failed", err, { jobId: job?.id });
    });

    this.worker.on("error", (err) => {
      Logger.error("Institution worker error", err);
    });
  }

  async shutdown(): Promise<void> {
    Logger.info("Shutting down institution worker");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startInstitutionWorker(): Promise<InstitutionWorker> {
  const worker = new InstitutionWorker();
  Logger.info("Institution worker started");
  return worker;
}

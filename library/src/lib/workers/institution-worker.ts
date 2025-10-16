/**
 * Institution Extraction Worker
 *
 * Processes jobs from the institution-extraction queue using BullMQ
 */

import { Job } from "bullmq";
import { BackgroundJobs } from "../job-queue";
import { processArxivInstitutionsAsync } from "../arxiv-auto-import";
import { Logger } from "../logging/logger";
import { BaseWorker } from "./base-worker";

export class InstitutionWorker extends BaseWorker<
  BackgroundJobs["extract-institutions"]
> {
  constructor() {
    super({
      queueName: "institution-extraction",
      concurrency: 2, // Process 2 jobs concurrently
      maxJobsPerMinute: 10, // Rate limiting for external APIs
    });
  }

  protected async processJob(
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
}

// Export for standalone usage
export async function startInstitutionWorker(): Promise<InstitutionWorker> {
  const worker = new InstitutionWorker();
  Logger.info("Institution worker started");
  return worker;
}

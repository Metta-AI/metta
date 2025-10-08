/**
 * LLM Processing Worker
 *
 * Processes jobs from the llm-processing queue using BullMQ
 */

import { Job } from "bullmq";
import { BackgroundJobs } from "../job-queue";
import { PaperAbstractService } from "../paper-abstract-service";
import { Logger } from "../logging/logger";
import { BaseWorker } from "./base-worker";

export class LLMWorker extends BaseWorker<
  BackgroundJobs["generate-llm-abstract"]
> {
  constructor() {
    super({
      queueName: "llm-processing",
      concurrency: 1, // LLM jobs are resource-intensive, process one at a time
      maxJobsPerMinute: 5, // OpenAI rate limiting
    });
  }

  protected async processJob(
    job: Job<BackgroundJobs["generate-llm-abstract"]>
  ): Promise<void> {
    const { paperId } = job.data;

    Logger.info("LLM Worker: Generating abstract for paper", { paperId });

    try {
      const abstract =
        await PaperAbstractService.generateAbstractForPaper(paperId);

      if (abstract) {
        Logger.info("LLM Worker: Generated abstract for paper", { paperId });
      } else {
        Logger.warn("LLM Worker: No abstract generated for paper", { paperId });
      }
    } catch (error) {
      Logger.error(
        "LLM Worker: Failed to generate abstract",
        error instanceof Error ? error : new Error(String(error)),
        { paperId }
      );
      throw error; // BullMQ will handle retries
    }
  }
}

// Export for standalone usage
export async function startLLMWorker(): Promise<LLMWorker> {
  const worker = new LLMWorker();
  Logger.info("LLM worker started");
  return worker;
}

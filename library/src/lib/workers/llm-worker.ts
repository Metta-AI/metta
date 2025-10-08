/**
 * LLM Processing Worker
 *
 * Processes jobs from the llm-processing queue using BullMQ
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { PaperAbstractService } from "../paper-abstract-service";
import { Logger } from "../logging/logger";

export class LLMWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      "llm-processing",
      async (job: Job<BackgroundJobs["generate-llm-abstract"]>) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: 1, // LLM jobs are resource-intensive, process one at a time
        limiter: {
          max: 5, // Max 5 LLM jobs
          duration: 60000, // Per minute (OpenAI rate limiting)
        },
      }
    );

    this.setupEventHandlers();
  }

  private async processJob(
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

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      Logger.debug("LLM job completed", { jobId: job.id });
    });

    this.worker.on("failed", (job, err) => {
      Logger.error("LLM job failed", err, { jobId: job?.id });
    });

    this.worker.on("error", (err) => {
      Logger.error("LLM worker error", err);
    });
  }

  async shutdown(): Promise<void> {
    Logger.info("Shutting down LLM worker");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startLLMWorker(): Promise<LLMWorker> {
  const worker = new LLMWorker();
  Logger.info("LLM worker started");
  return worker;
}

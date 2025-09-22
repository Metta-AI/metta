/**
 * LLM Processing Worker
 *
 * Processes jobs from the llm-processing queue using BullMQ
 */

import { Worker, Job } from "bullmq";
import { redisConfig, BackgroundJobs } from "../job-queue";
import { PaperAbstractService } from "../paper-abstract-service";

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

    console.log(`🤖 [LLM Worker] Generating abstract for paper ${paperId}`);

    try {
      const abstract =
        await PaperAbstractService.generateAbstractForPaper(paperId);

      if (abstract) {
        console.log(`✅ [LLM Worker] Generated abstract for paper ${paperId}`);
      } else {
        console.log(
          `⚠️ [LLM Worker] No abstract generated for paper ${paperId}`
        );
      }
    } catch (error) {
      console.error(
        `❌ [LLM Worker] Failed to generate abstract for paper ${paperId}:`,
        error
      );
      throw error; // BullMQ will handle retries
    }
  }

  private setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      console.log(`✅ LLM job ${job.id} completed successfully`);
    });

    this.worker.on("failed", (job, err) => {
      console.error(`❌ LLM job ${job?.id} failed:`, err.message);
    });

    this.worker.on("error", (err) => {
      console.error("🚨 LLM worker error:", err);
    });
  }

  async shutdown(): Promise<void> {
    console.log("🛑 Shutting down LLM worker...");
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startLLMWorker(): Promise<LLMWorker> {
  const worker = new LLMWorker();
  console.log("🚀 LLM worker started");
  return worker;
}

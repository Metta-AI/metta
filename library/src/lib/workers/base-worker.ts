/**
 * Base Worker Class
 *
 * Abstract base class for BullMQ workers that provides:
 * - Common setup and teardown logic
 * - Standardized event handling
 * - Consistent logging patterns
 * - Type-safe job processing
 */

import { Worker, Job, WorkerOptions } from "bullmq";
import { redisConfig } from "../job-queue";
import { Logger } from "../logging/logger";

export interface BaseWorkerConfig {
  queueName: string;
  concurrency?: number;
  maxJobsPerMinute?: number;
}

export abstract class BaseWorker<TJobData = any> {
  protected worker: Worker;
  protected readonly workerName: string;

  constructor(config: BaseWorkerConfig) {
    this.workerName = config.queueName;

    const workerOptions: WorkerOptions = {
      connection: redisConfig,
      concurrency: config.concurrency ?? 1,
    };

    // Add rate limiter if specified
    if (config.maxJobsPerMinute) {
      workerOptions.limiter = {
        max: config.maxJobsPerMinute,
        duration: 60000, // Per minute
      };
    }

    this.worker = new Worker(
      config.queueName,
      async (job: Job<TJobData>) => {
        return await this.processJob(job);
      },
      workerOptions
    );

    this.setupEventHandlers();
  }

  /**
   * Abstract method that subclasses must implement to process jobs
   */
  protected abstract processJob(job: Job<TJobData>): Promise<void>;

  /**
   * Setup standard event handlers for worker lifecycle
   * Can be overridden by subclasses for custom behavior
   */
  protected setupEventHandlers(): void {
    this.worker.on("completed", (job) => {
      Logger.debug(`${this.workerName} job completed`, { jobId: job.id });
    });

    this.worker.on("failed", (job, err) => {
      Logger.error(`${this.workerName} job failed`, err, { jobId: job?.id });
    });

    this.worker.on("error", (err) => {
      Logger.error(`${this.workerName} worker error`, err);
    });
  }

  /**
   * Gracefully shutdown the worker
   */
  async shutdown(): Promise<void> {
    Logger.info(`Shutting down ${this.workerName} worker`);
    await this.worker.close();
  }

  /**
   * Get the underlying BullMQ Worker instance
   */
  getWorker(): Worker {
    return this.worker;
  }
}

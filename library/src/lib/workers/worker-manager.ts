/**
 * Worker Manager - Starts and manages all background workers
 *
 * Run this file to start all workers for local development:
 * tsx src/lib/workers/worker-manager.ts
 */

import { InstitutionWorker } from "./institution-worker";
import { LLMWorker } from "./llm-worker";
import { TaggingWorker } from "./tagging-worker";
import { ExternalNotificationWorker } from "./external-notification-worker";
import { JobQueueService } from "../job-queue";
import { Logger } from "../logging/logger";

export class WorkerManager {
  private institutionWorker?: InstitutionWorker;
  private llmWorker?: LLMWorker;
  private taggingWorker?: TaggingWorker;
  private externalNotificationWorker?: ExternalNotificationWorker;
  private isShuttingDown = false;

  async start(): Promise<void> {
    Logger.info("Starting all background workers");

    try {
      // Start all workers
      this.institutionWorker = new InstitutionWorker();
      this.llmWorker = new LLMWorker();
      this.taggingWorker = new TaggingWorker();
      this.externalNotificationWorker = new ExternalNotificationWorker();

      Logger.info("All workers started successfully", {
        workers: [
          "Institution Worker (concurrency: 2)",
          "LLM Worker (concurrency: 1)",
          "Tagging Worker (concurrency: 3)",
          "External Notification Worker (concurrency: 3)",
        ],
      });

      // Setup graceful shutdown
      this.setupGracefulShutdown();

      // Display queue stats every 30 seconds
      this.startStatsReporting();
    } catch (error) {
      const errorInstance =
        error instanceof Error ? error : new Error(String(error));
      Logger.error("Failed to start workers", errorInstance);
      await this.shutdown();
      process.exit(1);
    }
  }

  private setupGracefulShutdown(): void {
    process.on("SIGINT", () => this.handleShutdown("SIGINT"));
    process.on("SIGTERM", () => this.handleShutdown("SIGTERM"));
    process.on("uncaughtException", (error) => {
      Logger.error("Uncaught exception", error);
      this.handleShutdown("uncaughtException");
    });
  }

  private async handleShutdown(signal: string): Promise<void> {
    if (this.isShuttingDown) return;
    this.isShuttingDown = true;

    Logger.info("Shutting down gracefully", { signal });
    await this.shutdown();
    process.exit(0);
  }

  private async startStatsReporting(): Promise<void> {
    setInterval(async () => {
      if (this.isShuttingDown) return;

      try {
        const stats = await JobQueueService.getQueueStats();
        Logger.debug("Queue Statistics", { stats });
      } catch (error) {
        Logger.warn("Failed to get queue stats", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }, 30000); // Every 30 seconds
  }

  async shutdown(): Promise<void> {
    Logger.info("Shutting down all workers");

    const shutdownPromises = [];

    if (this.institutionWorker) {
      shutdownPromises.push(this.institutionWorker.shutdown());
    }
    if (this.llmWorker) {
      shutdownPromises.push(this.llmWorker.shutdown());
    }
    if (this.taggingWorker) {
      shutdownPromises.push(this.taggingWorker.shutdown());
    }
    if (this.externalNotificationWorker) {
      shutdownPromises.push(this.externalNotificationWorker.shutdown());
    }

    await Promise.all(shutdownPromises);
    await JobQueueService.shutdown();

    Logger.info("All workers shut down successfully");
  }
}

// Run if called directly
if (require.main === module) {
  const manager = new WorkerManager();
  manager.start().catch((error) => {
    const errorInstance =
      error instanceof Error ? error : new Error(String(error));
    Logger.error("Failed to start worker manager", errorInstance);
    process.exit(1);
  });
}

export default WorkerManager;

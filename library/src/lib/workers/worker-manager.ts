/**
 * Worker Manager - Starts and manages all background workers
 *
 * Run this file to start all workers for local development:
 * tsx src/lib/workers/worker-manager.ts
 */

import { InstitutionWorker } from "./institution-worker";
import { LLMWorker } from "./llm-worker";
import { TaggingWorker } from "./tagging-worker";
import { JobQueueService } from "../job-queue";

export class WorkerManager {
  private institutionWorker?: InstitutionWorker;
  private llmWorker?: LLMWorker;
  private taggingWorker?: TaggingWorker;
  private isShuttingDown = false;

  async start(): Promise<void> {
    console.log("🚀 Starting all background workers...");

    try {
      // Start all workers
      this.institutionWorker = new InstitutionWorker();
      this.llmWorker = new LLMWorker();
      this.taggingWorker = new TaggingWorker();

      console.log("✅ All workers started successfully!");
      console.log("📊 Worker status:");
      console.log("   🏛️ Institution Worker: Running (concurrency: 2)");
      console.log("   🤖 LLM Worker: Running (concurrency: 1)");
      console.log("   🏷️ Tagging Worker: Running (concurrency: 3)");

      // Setup graceful shutdown
      this.setupGracefulShutdown();

      // Display queue stats every 30 seconds
      this.startStatsReporting();
    } catch (error) {
      console.error("❌ Failed to start workers:", error);
      await this.shutdown();
      process.exit(1);
    }
  }

  private setupGracefulShutdown(): void {
    process.on("SIGINT", () => this.handleShutdown("SIGINT"));
    process.on("SIGTERM", () => this.handleShutdown("SIGTERM"));
    process.on("uncaughtException", (error) => {
      console.error("🚨 Uncaught exception:", error);
      this.handleShutdown("uncaughtException");
    });
  }

  private async handleShutdown(signal: string): Promise<void> {
    if (this.isShuttingDown) return;
    this.isShuttingDown = true;

    console.log(`\n🛑 Received ${signal}, shutting down gracefully...`);
    await this.shutdown();
    process.exit(0);
  }

  private async startStatsReporting(): Promise<void> {
    setInterval(async () => {
      if (this.isShuttingDown) return;

      try {
        const stats = await JobQueueService.getQueueStats();
        console.log("\n📊 Queue Statistics:");
        console.log(
          `   🏛️ Institution: ${stats.institution.active} active, ${stats.institution.waiting} waiting, ${stats.institution.completed} completed`
        );
        console.log(
          `   🤖 LLM: ${stats.llm.active} active, ${stats.llm.waiting} waiting, ${stats.llm.completed} completed`
        );
        console.log(
          `   🏷️ Tagging: ${stats.tagging.active} active, ${stats.tagging.waiting} waiting, ${stats.tagging.completed} completed`
        );
      } catch (error) {
        console.error("❌ Failed to get queue stats:", error);
      }
    }, 30000); // Every 30 seconds
  }

  async shutdown(): Promise<void> {
    console.log("🛑 Shutting down all workers...");

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

    await Promise.all(shutdownPromises);
    await JobQueueService.shutdown();

    console.log("✅ All workers shut down successfully");
  }
}

// Run if called directly
if (require.main === module) {
  const manager = new WorkerManager();
  manager.start().catch((error) => {
    console.error("💥 Failed to start worker manager:", error);
    process.exit(1);
  });
}

export default WorkerManager;

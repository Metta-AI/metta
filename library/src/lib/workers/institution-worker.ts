/**
 * Institution Extraction Worker
 * 
 * Processes jobs from the institution-extraction queue using BullMQ
 */

import { Worker, Job } from 'bullmq';
import { redisConfig, BackgroundJobs } from '../job-queue';
import { processArxivInstitutionsAsync } from '../arxiv-auto-import';

export class InstitutionWorker {
  private worker: Worker;

  constructor() {
    this.worker = new Worker(
      'institution-extraction',
      async (job: Job<BackgroundJobs['extract-institutions']>) => {
        return await this.processJob(job);
      },
      {
        connection: redisConfig,
        concurrency: 2, // Process 2 jobs concurrently
        limiter: {
          max: 10,      // Max 10 jobs
          duration: 60000, // Per minute (rate limiting for external APIs)
        },
      }
    );

    this.setupEventHandlers();
  }

  private async processJob(job: Job<BackgroundJobs['extract-institutions']>): Promise<void> {
    const { paperId, arxivUrl } = job.data;
    
    console.log(`🏛️ [Institution Worker] Processing paper ${paperId}`);
    console.log(`   arXiv URL: ${arxivUrl}`);

    try {
      await processArxivInstitutionsAsync(paperId, arxivUrl);
      console.log(`✅ [Institution Worker] Completed paper ${paperId}`);
    } catch (error) {
      console.error(`❌ [Institution Worker] Failed paper ${paperId}:`, error);
      throw error; // BullMQ will handle retries
    }
  }

  private setupEventHandlers(): void {
    this.worker.on('completed', (job) => {
      console.log(`✅ Institution job ${job.id} completed successfully`);
    });

    this.worker.on('failed', (job, err) => {
      console.error(`❌ Institution job ${job?.id} failed:`, err.message);
    });

    this.worker.on('error', (err) => {
      console.error('🚨 Institution worker error:', err);
    });
  }

  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down institution worker...');
    await this.worker.close();
  }
}

// Export for standalone usage
export async function startInstitutionWorker(): Promise<InstitutionWorker> {
  const worker = new InstitutionWorker();
  console.log('🚀 Institution worker started');
  return worker;
}


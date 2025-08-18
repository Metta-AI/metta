/**
 * Background Job Processing for arXiv Auto-Import
 *
 * This module handles asynchronous processing of arXiv papers to avoid blocking
 * post creation. Jobs are processed in the background with proper error handling.
 */

import { prisma } from "@/lib/db/prisma";
import { processArxivInstitutionsAsync } from "./arxiv-auto-import";

export interface ArxivInstitutionJob {
  id: string;
  paperId: string;
  arxivUrl: string;
  status: "pending" | "processing" | "completed" | "failed";
  createdAt: Date;
  processedAt?: Date;
  error?: string;
}

/**
 * Queue an arXiv institution processing job for background execution
 */
export async function queueArxivInstitutionProcessing(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  console.log(`📤 Queuing institution processing for paper ${paperId}`);

  // For now, just process directly in background
  // In a production app, you'd use a proper job queue like Bull/BullMQ
  setImmediate(async () => {
    try {
      await processArxivInstitutionsAsync(paperId, arxivUrl);
    } catch (error) {
      console.error(
        `❌ Background institution job failed for paper ${paperId}:`,
        error
      );
    }
  });
}

/**
 * Process a single arXiv institution job with error handling and status tracking
 */
export async function processArxivInstitutionJob(
  job: ArxivInstitutionJob
): Promise<void> {
  console.log(`🔄 Processing institution job for paper ${job.paperId}`);

  try {
    await processArxivInstitutionsAsync(job.paperId, job.arxivUrl);
    console.log(`✅ Completed institution job for paper ${job.paperId}`);
  } catch (error) {
    console.error(`❌ Failed institution job for paper ${job.paperId}:`, error);
    throw error;
  }
}

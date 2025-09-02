/**
 * Background Job Processing using BullMQ
 *
 * This module provides a unified interface for queueing background jobs
 * using BullMQ for reliable processing with retries and monitoring.
 */

import { JobQueueService } from './job-queue';

/**
 * Queue an arXiv institution processing job for background execution
 */
export async function queueArxivInstitutionProcessing(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  await JobQueueService.queueInstitutionExtraction(paperId, arxivUrl);
}

/**
 * Queue author extraction for a paper
 */
export async function queueArxivAuthorProcessing(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  await JobQueueService.queueAuthorExtraction(paperId, arxivUrl);
}

/**
 * Queue LLM abstract generation for a paper
 */
export async function queueLLMAbstractGeneration(
  paperId: string
): Promise<void> {
  await JobQueueService.queueLLMAbstractGeneration(paperId);
}

/**
 * Queue auto-tagging for a paper
 */
export async function queueAutoTagging(
  paperId: string
): Promise<void> {
  await JobQueueService.queueAutoTagging(paperId);
}

/**
 * Get background job statistics
 */
export async function getBackgroundJobStats() {
  return await JobQueueService.getQueueStats();
}

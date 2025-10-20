import { prisma } from "@/lib/db/prisma";
import {
  generateLLMAbstract,
  LLMAbstract,
} from "./llm-abstract-generator-clean";
import { Logger } from "./logging/logger";

/**
 * Service for generating and managing LLM-enhanced abstracts for papers
 */
export class PaperAbstractService {
  /**
   * Generate LLM abstract for a paper by its ID
   */
  static async generateAbstractForPaper(
    paperId: string
  ): Promise<LLMAbstract | null> {
    try {
      Logger.info("Generating LLM abstract for paper", { paperId });

      // Fetch paper from database
      const paper = await prisma.paper.findUnique({
        where: { id: paperId },
        select: {
          id: true,
          title: true,
          link: true,
          source: true,
          externalId: true,
          llmAbstract: true,
          llmAbstractGeneratedAt: true,
        },
      });

      if (!paper) {
        Logger.warn("Paper not found", { paperId });
        return null;
      }

      if (!paper.link) {
        Logger.warn("No PDF link found for paper", { paperId });
        return null;
      }

      // Check if we already have an LLM abstract
      if (paper.llmAbstract && paper.llmAbstractGeneratedAt) {
        const existingAbstract = paper.llmAbstract as unknown as LLMAbstract;
        Logger.debug("Found existing LLM abstract for paper", { paperId });
        return existingAbstract;
      }

      // Generate new LLM abstract
      return await this.generateNewAbstract(paper);
    } catch (error) {
      Logger.error(
        "Error generating abstract for paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId }
      );
      return null;
    }
  }

  /**
   * Generate abstracts for multiple papers in batch
   */
  static async generateAbstractsForPapers(
    paperIds: string[]
  ): Promise<Map<string, LLMAbstract | null>> {
    const results = new Map<string, LLMAbstract | null>();

    Logger.info("Generating LLM abstracts for batch", {
      paperCount: paperIds.length,
    });

    // Process papers sequentially to avoid overwhelming the API
    for (const paperId of paperIds) {
      const abstract = await this.generateAbstractForPaper(paperId);
      results.set(paperId, abstract);

      // Small delay to be nice to the API
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    Logger.info("Completed batch abstract generation", {
      paperCount: paperIds.length,
    });
    return results;
  }

  /**
   * Generate abstract for papers that don't have one yet
   */
  static async generateMissingAbstracts(limit: number = 10): Promise<void> {
    try {
      // Find papers without LLM abstracts that have PDF links
      const papersNeedingAbstracts = await prisma.paper.findMany({
        where: {
          llmAbstract: null as any,
          link: { not: null },
        },
        select: { id: true },
        take: limit,
        orderBy: { createdAt: "desc" }, // Process newest papers first
      });

      if (papersNeedingAbstracts.length === 0) {
        Logger.info("No papers found that need LLM abstracts");
        return;
      }

      Logger.info("Found papers needing LLM abstracts", {
        count: papersNeedingAbstracts.length,
      });

      const paperIds = papersNeedingAbstracts.map((p) => p.id);
      await this.generateAbstractsForPapers(paperIds);
    } catch (error) {
      Logger.error(
        "Error generating missing abstracts",
        error instanceof Error ? error : new Error(String(error))
      );
    }
  }

  /**
   * Generate new abstract for a paper
   */
  private static async generateNewAbstract(
    paper: any
  ): Promise<LLMAbstract | null> {
    try {
      // Normalize URL to ensure we get the actual PDF
      const normalizedUrl = this.normalizePdfUrl(paper.link);
      Logger.info(`📥 Fetching PDF from: ${normalizedUrl}`);

      const response = await fetch(normalizedUrl, {
        headers: {
          Accept: "application/pdf,*/*",
          "User-Agent": "Mozilla/5.0 (compatible; LibraryBot/1.0)",
        },
        redirect: "follow",
      });

      if (!response.ok) {
        const error = new Error(
          `Failed to fetch PDF: ${response.status} ${response.statusText}`
        );
        Logger.error(`❌ Failed to fetch PDF`, error, {
          status: response.status,
          statusText: response.statusText,
        });
        return null;
      }

      // Validate content type
      const contentType = response.headers.get("content-type") || "";
      Logger.info(`📄 Response content-type: ${contentType}`);

      if (contentType.includes("text/html")) {
        const error = new Error("URL returned HTML instead of PDF");
        Logger.error(`❌ URL returned HTML instead of PDF`, error, {
          normalizedUrl,
          originalUrl: paper.link,
        });
        return null;
      }

      const pdfBuffer = Buffer.from(await response.arrayBuffer());
      Logger.info(`📄 Successfully fetched PDF (${pdfBuffer.length} bytes)`);

      // Generate LLM abstract directly (no separate content extraction step)
      const homepageUrl = this.getHomepageUrl(paper);
      const llmAbstract = await generateLLMAbstract(
        paper.title,
        undefined, // pdfContent is optional, we use pdfBuffer for enhanced extraction
        paper.link,
        homepageUrl,
        pdfBuffer
      );

      // Save to database
      await this.saveAbstractToDatabase(paper.id, llmAbstract);

      Logger.info(`✅ Generated and saved LLM abstract for paper: ${paper.id}`);
      return llmAbstract;
    } catch (error) {
      Logger.error(
        `❌ Error generating new abstract for paper ${paper.id}:`,
        error
      );
      return null;
    }
  }

  /**
   * Save LLM abstract to database
   */
  private static async saveAbstractToDatabase(
    paperId: string,
    llmAbstract: LLMAbstract
  ): Promise<void> {
    try {
      await prisma.paper.update({
        where: { id: paperId },
        data: {
          llmAbstract: llmAbstract as any, // Prisma will handle JSON serialization
          llmAbstractGeneratedAt: new Date(),
        },
      });

      Logger.info(`💾 Saved LLM abstract to database for paper: ${paperId}`);
    } catch (error) {
      Logger.error(`❌ Error saving LLM abstract to database:`, error);
      throw error;
    }
  }

  /**
   * Convert various paper URLs to direct PDF URLs
   */
  private static normalizePdfUrl(url: string): string {
    // Handle arXiv URLs
    if (url.includes("arxiv.org")) {
      // Convert arXiv abstract URL to PDF URL
      const arxivIdMatch = url.match(/arxiv\.org\/abs\/(.+)/);
      if (arxivIdMatch) {
        return `https://arxiv.org/pdf/${arxivIdMatch[1]}.pdf`;
      }

      // If it's already a PDF URL, use as-is
      if (url.includes("/pdf/")) {
        return url;
      }
    }

    // For other URLs, assume they're already direct PDF links
    return url;
  }

  /**
   * Generate homepage URL from paper metadata
   */
  private static getHomepageUrl(paper: any): string | undefined {
    if (paper.source === "arxiv" && paper.externalId) {
      return `https://arxiv.org/abs/${paper.externalId}`;
    }

    // For other sources, the link might be the homepage
    if (paper.link && !paper.link.includes(".pdf")) {
      return paper.link;
    }

    return undefined;
  }

  /**
   * Get LLM abstract for a paper (from database)
   */
  static async getAbstractForPaper(
    paperId: string
  ): Promise<LLMAbstract | null> {
    try {
      const paper = await prisma.paper.findUnique({
        where: { id: paperId },
        select: {
          llmAbstract: true,
          llmAbstractGeneratedAt: true,
        },
      });

      if (paper?.llmAbstract) {
        return paper.llmAbstract as unknown as LLMAbstract;
      }

      return null;
    } catch (error) {
      Logger.error(
        `❌ Error fetching LLM abstract for paper ${paperId}:`,
        error
      );
      return null;
    }
  }

  /**
   * Delete LLM abstract for a paper (useful for regeneration)
   */
  static async deleteAbstractForPaper(paperId: string): Promise<void> {
    try {
      await prisma.paper.update({
        where: { id: paperId },
        data: {
          llmAbstract: null as any,
          llmAbstractGeneratedAt: null,
        },
      });

      Logger.info(`🗑️ Deleted LLM abstract for paper: ${paperId}`);
    } catch (error) {
      Logger.error(
        `❌ Error deleting LLM abstract for paper ${paperId}:`,
        error
      );
      throw error;
    }
  }
}

import { prisma } from "@/lib/db/prisma";
import { extractPdfContentWithImages } from "./pdf-content-extractor";
import {
  generateLLMAbstract,
  updateLLMAbstractIfNeeded,
  LLMAbstract,
} from "./llm-abstract-generator-clean";

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
      console.log(`📋 Generating LLM abstract for paper: ${paperId}`);

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
        console.error(`❌ Paper not found: ${paperId}`);
        return null;
      }

      if (!paper.link) {
        console.error(`❌ No PDF link found for paper: ${paperId}`);
        return null;
      }

      // Check if we already have an LLM abstract
      if (paper.llmAbstract && paper.llmAbstractGeneratedAt) {
        const existingAbstract = paper.llmAbstract as unknown as LLMAbstract;
        console.log(`📋 Found existing LLM abstract for paper: ${paperId}`);
        return existingAbstract;
      }

      // Generate new LLM abstract
      return await this.generateNewAbstract(paper);
    } catch (error) {
      console.error(
        `❌ Error generating abstract for paper ${paperId}:`,
        error
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

    console.log(`📋 Generating LLM abstracts for ${paperIds.length} papers...`);

    // Process papers sequentially to avoid overwhelming the API
    for (const paperId of paperIds) {
      const abstract = await this.generateAbstractForPaper(paperId);
      results.set(paperId, abstract);

      // Small delay to be nice to the API
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    console.log(
      `✅ Completed batch abstract generation for ${paperIds.length} papers`
    );
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
        console.log("✅ No papers found that need LLM abstracts");
        return;
      }

      console.log(
        `📋 Found ${papersNeedingAbstracts.length} papers needing LLM abstracts`
      );

      const paperIds = papersNeedingAbstracts.map((p) => p.id);
      await this.generateAbstractsForPapers(paperIds);
    } catch (error) {
      console.error("❌ Error generating missing abstracts:", error);
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
      console.log(`📥 Fetching PDF from: ${normalizedUrl}`);

      const response = await fetch(normalizedUrl, {
        headers: {
          Accept: "application/pdf,*/*",
          "User-Agent": "Mozilla/5.0 (compatible; LibraryBot/1.0)",
        },
        redirect: "follow",
      });

      if (!response.ok) {
        console.error(
          `❌ Failed to fetch PDF: ${response.status} ${response.statusText}`
        );
        return null;
      }

      // Validate content type
      const contentType = response.headers.get("content-type") || "";
      console.log(`📄 Response content-type: ${contentType}`);

      if (contentType.includes("text/html")) {
        console.error(
          `❌ URL returned HTML instead of PDF. URL might be incorrect: ${normalizedUrl}`
        );
        console.error(`❌ Original URL: ${paper.link}`);
        return null;
      }

      const pdfBuffer = Buffer.from(await response.arrayBuffer());
      console.log(`📄 Successfully fetched PDF (${pdfBuffer.length} bytes)`);

      // Generate LLM abstract directly (no separate content extraction step)
      const homepageUrl = this.getHomepageUrl(paper);
      const llmAbstract = await generateLLMAbstract(
        paper.title,
        {} as any, // Empty pdfContent since we're going direct to enhanced extraction
        paper.link,
        homepageUrl,
        pdfBuffer
      );

      // Save to database
      await this.saveAbstractToDatabase(paper.id, llmAbstract);

      console.log(`✅ Generated and saved LLM abstract for paper: ${paper.id}`);
      return llmAbstract;
    } catch (error) {
      console.error(
        `❌ Error generating new abstract for paper ${paper.id}:`,
        error
      );
      return null;
    }
  }

  /**
   * Fetch PDF and extract content
   */
  private static async fetchAndExtractPdf(pdfUrl: string): Promise<any> {
    try {
      console.log(`📄 Fetching PDF from: ${pdfUrl}`);

      // Handle arXiv URLs - convert to PDF URL if needed
      const finalPdfUrl = this.normalizePdfUrl(pdfUrl);

      const response = await fetch(finalPdfUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (compatible; paper-abstract-service/1.0)",
        },
      });

      if (!response.ok) {
        console.error(
          `❌ Failed to fetch PDF: ${response.status} ${response.statusText}`
        );
        return null;
      }

      const pdfBuffer = Buffer.from(await response.arrayBuffer());
      console.log(`📄 Successfully fetched PDF (${pdfBuffer.length} bytes)`);

      // Extract content from PDF (including images)
      const pdfContent = await extractPdfContentWithImages(pdfBuffer);
      console.log(
        `📄 Extracted PDF content: ${pdfContent.pageCount} pages, ${pdfContent.figuresWithImages.length} figures with images`
      );

      // Attach the PDF buffer for reuse in enhanced abstract generation
      (pdfContent as any)._pdfBuffer = pdfBuffer;

      return pdfContent;
    } catch (error) {
      console.error(`❌ Error fetching/extracting PDF:`, error);
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

      console.log(`💾 Saved LLM abstract to database for paper: ${paperId}`);
    } catch (error) {
      console.error(`❌ Error saving LLM abstract to database:`, error);
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
      console.error(
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

      console.log(`🗑️ Deleted LLM abstract for paper: ${paperId}`);
    } catch (error) {
      console.error(
        `❌ Error deleting LLM abstract for paper ${paperId}:`,
        error
      );
      throw error;
    }
  }
}

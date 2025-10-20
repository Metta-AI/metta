import { generateObject } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";
import { TagVocabularyService } from "./tag-vocabulary-service";
import { prisma } from "@/lib/db/prisma";
import { config } from "./config";
import { Logger } from "./logging/logger";

/**
 * Schema for OpenAI tag suggestion response
 */
const TagSuggestionSchema = z.object({
  suggestedTags: z
    .array(z.string())
    .describe("Array of suggested tags from the provided vocabulary"),
  reasoning: z
    .string()
    .describe("Brief explanation of why these tags were selected"),
});

/**
 * Service for automatically tagging papers using OpenAI vision
 */
export class AutoTaggingService {
  /**
   * Auto-tag a paper using its PDF content
   */
  static async autoTagPaper(paperId: string): Promise<string[] | null> {
    try {
      Logger.info("Auto-tagging paper", { paperId });

      // Get paper data
      const paper = await prisma.paper.findUnique({
        where: { id: paperId },
        select: {
          id: true,
          title: true,
          abstract: true,
          link: true,
          source: true,
          externalId: true,
          tags: true,
        },
      });

      if (!paper) {
        Logger.warn("Paper not found for auto-tagging", { paperId });
        return null;
      }

      // Skip if paper already has tags
      if (paper.tags && paper.tags.length > 0) {
        Logger.debug("Paper already has tags, skipping", {
          paperId,
          tags: paper.tags,
        });
        return paper.tags;
      }

      // Get tag vocabulary
      const tagVocabulary = await TagVocabularyService.getAllTags();
      if (tagVocabulary.length === 0) {
        Logger.warn("No tag vocabulary available for auto-tagging");
        return null;
      }

      // Generate tags based on available content
      let suggestedTags: string[] = [];

      // Try PDF analysis first (if available)
      if (
        paper.link &&
        this.canAnalyzePdf({ link: paper.link, source: paper.source })
      ) {
        suggestedTags = await this.generateTagsFromPdf(
          { ...paper, link: paper.link },
          tagVocabulary
        );
      }

      // Fallback to text-based analysis
      if (suggestedTags.length === 0 && (paper.title || paper.abstract)) {
        suggestedTags = await this.generateTagsFromText(paper, tagVocabulary);
      }

      if (suggestedTags.length > 0) {
        // Update paper with suggested tags
        await prisma.paper.update({
          where: { id: paperId },
          data: { tags: suggestedTags },
        });

        Logger.info("Auto-tagged paper", { paperId, tags: suggestedTags });
        return suggestedTags;
      } else {
        Logger.warn("No suitable tags found for paper", { paperId });
        return null;
      }
    } catch (error) {
      Logger.error(
        "Error auto-tagging paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId }
      );
      return null;
    }
  }

  /**
   * Normalize URL to get the actual PDF, not HTML pages
   */
  private static normalizePdfUrl(url: string): string {
    if (!url) return url;

    // arXiv URLs: convert from abstract page to PDF
    if (url.includes("arxiv.org/abs/")) {
      const normalizedUrl = url.replace("/abs/", "/pdf/") + ".pdf";
      Logger.debug("Converted arXiv abstract URL to PDF", { normalizedUrl });
      return normalizedUrl;
    }

    // Other common patterns can be added here
    // For now, return the original URL
    return url;
  }

  /**
   * Generate tags from PDF content using OpenAI vision
   */
  private static async generateTagsFromPdf(
    paper: { title: string; abstract?: string | null; link: string },
    tagVocabulary: string[]
  ): Promise<string[]> {
    try {
      if (!config.llm.anthropicApiKey) {
        Logger.warn("Anthropic API key not available for PDF analysis");
        return [];
      }

      Logger.info("Analyzing PDF for auto-tagging", {
        paperTitle: paper.title,
      });

      // Normalize URL to get actual PDF
      const normalizedUrl = this.normalizePdfUrl(paper.link);
      Logger.debug("Fetching PDF", { url: normalizedUrl });

      // Fetch PDF
      const pdfResponse = await fetch(normalizedUrl, {
        headers: {
          Accept: "application/pdf,*/*",
          "User-Agent": "Mozilla/5.0 (compatible; LibraryBot/1.0)",
        },
        redirect: "follow",
      });
      if (!pdfResponse.ok) {
        Logger.warn("Could not fetch PDF", {
          status: pdfResponse.status,
          url: normalizedUrl,
        });
        return [];
      }

      const contentType = pdfResponse.headers.get("content-type") || "";
      Logger.debug("PDF response received", { contentType });

      if (contentType.includes("text/html")) {
        Logger.warn("URL returned HTML instead of PDF", {
          normalizedUrl,
          originalUrl: paper.link,
        });
        return [];
      }

      const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
      Logger.debug("PDF buffer loaded", { size: pdfBuffer.length });

      // Validate PDF header
      const pdfHeader = pdfBuffer.slice(0, 8).toString();
      if (!pdfHeader.startsWith("%PDF")) {
        Logger.warn("Invalid PDF header", {
          pdfHeader,
          firstBytes: pdfBuffer.slice(0, 50).toString("hex"),
        });
        return [];
      }

      // Check file size (Anthropic has limits)
      const maxSize = 512 * 1024 * 1024; // 512MB limit for Anthropic
      if (pdfBuffer.length > maxSize) {
        Logger.warn("PDF too large for analysis", {
          size: pdfBuffer.length,
          maxSize,
        });
        return [];
      }

      const tagVocabularyText = tagVocabulary.join(", ");

      Logger.info("Sending PDF to Anthropic for analysis", {
        paperTitle: paper.title,
      });
      const result = await generateObject({
        model: anthropic("claude-3-5-sonnet-20241022"),
        schema: TagSuggestionSchema,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `Analyze this research paper and suggest 3-5 relevant tags from the provided vocabulary.

Paper Title: "${paper.title}"
${paper.abstract ? `Abstract: "${paper.abstract}"` : ""}

Available Tag Vocabulary:
${tagVocabularyText}

Instructions:
1. Analyze the paper's content, methodology, and research domain
2. Select 3-5 most relevant tags from the vocabulary above
3. Only use tags that exist in the provided vocabulary
4. Focus on the paper's main contributions, methods, and research areas
5. Provide brief reasoning for your selections

IMPORTANT: Only return tags that appear EXACTLY in the vocabulary list above.`,
              },
              {
                type: "file",
                data: pdfBuffer,
                mediaType: "application/pdf",
              },
            ],
          },
        ],
        maxRetries: 1,
      });

      // Validate suggested tags against vocabulary
      const validTags = result.object.suggestedTags.filter((tag) =>
        tagVocabulary.includes(tag)
      );

      if (validTags.length > 0) {
        Logger.info(`🔍 PDF analysis suggested: [${validTags.join(", ")}]`);
        Logger.info(`💭 Reasoning: ${result.object.reasoning}`);
      }

      return validTags;
    } catch (error) {
      Logger.error(`❌ Error in PDF analysis:`, error);

      // Log more details about the error
      if (error && typeof error === "object") {
        Logger.error(`❌ Error details:`, {
          message: (error as any).message,
          statusCode: (error as any).statusCode,
          responseBody: (error as any).responseBody,
          url: (error as any).url,
          paperTitle: paper.title,
          paperLink: paper.link,
        });
      }

      return [];
    }
  }

  /**
   * Generate tags from title and abstract using OpenAI
   */
  private static async generateTagsFromText(
    paper: { title: string; abstract?: string | null },
    tagVocabulary: string[]
  ): Promise<string[]> {
    try {
      if (!config.llm.anthropicApiKey) {
        Logger.warn("⚠️ Anthropic API key not available for text analysis");
        return [];
      }

      Logger.info(`📝 Analyzing text for paper: ${paper.title}`);

      const tagVocabularyText = tagVocabulary.join(", ");
      const textContent = `Title: ${paper.title}\n${paper.abstract ? `Abstract: ${paper.abstract}` : ""}`;

      const result = await generateObject({
        model: anthropic("claude-3-haiku-20240307"), // Use faster, cheaper model for text-only analysis
        schema: TagSuggestionSchema,
        messages: [
          {
            role: "user",
            content: `Analyze this research paper and suggest 3-5 relevant tags from the provided vocabulary.

${textContent}

Available Tag Vocabulary:
${tagVocabularyText}

Instructions:
1. Analyze the paper's content, methodology, and research domain based on the title and abstract
2. Select 3-5 most relevant tags from the vocabulary above
3. Only use tags that exist in the provided vocabulary
4. Focus on the paper's main contributions, methods, and research areas
5. Provide brief reasoning for your selections

IMPORTANT: Only return tags that appear EXACTLY in the vocabulary list above.`,
          },
        ],
        maxRetries: 1,
      });

      // Validate suggested tags against vocabulary
      const validTags = result.object.suggestedTags.filter((tag) =>
        tagVocabulary.includes(tag)
      );

      if (validTags.length > 0) {
        Logger.info(`📝 Text analysis suggested: [${validTags.join(", ")}]`);
        Logger.info(`💭 Reasoning: ${result.object.reasoning}`);
      }

      return validTags;
    } catch (error) {
      Logger.error(`❌ Error in text analysis:`, error);
      return [];
    }
  }

  /**
   * Check if a paper's PDF can be analyzed
   */
  private static canAnalyzePdf(paper: {
    link: string;
    source?: string | null;
  }): boolean {
    if (!paper.link) return false;

    // Check if it's a PDF URL
    if (paper.link.endsWith(".pdf")) return true;

    // Check if it's an arXiv URL (we can construct PDF URL)
    if (paper.source === "arxiv" && paper.link.includes("arxiv.org/abs/"))
      return true;

    return false;
  }
}

import { prisma } from "@/lib/db/prisma";
import {
  extractArxivId,
  fetchArxivPaper,
} from "../../scripts/fetch-arxiv-paper";
import { extractInstitutionsFromPdf } from "./pdf-institution-extractor";
import { PaperAbstractService } from "./paper-abstract-service";
import { Logger } from "./logging/logger";

/**
 * Normalizes author name for consistent storage
 */
function normalizeAuthorName(name: string): string {
  return name.trim();
}

/**
 * Creates or finds an author record
 */
async function getOrCreateAuthor(authorName: string) {
  const normalizedName = normalizeAuthorName(authorName);

  // Check if author already exists
  let author = await prisma.author.findUnique({
    where: { name: normalizedName },
  });

  // Create author if doesn't exist
  if (!author) {
    author = await prisma.author.create({
      data: {
        name: normalizedName,
        email: null,
        institution: null,
      },
    });
    Logger.debug("Created author", { authorName: normalizedName });
  }

  return author;
}

/**
 * Creates paper-author relationship if it doesn't exist
 */
async function linkAuthorToPaper(authorId: string, paperId: string) {
  // Check if relationship already exists
  const existingLink = await prisma.paperAuthor.findUnique({
    where: {
      paperId_authorId: {
        paperId,
        authorId,
      },
    },
  });

  if (!existingLink) {
    await prisma.paperAuthor.create({
      data: {
        paperId,
        authorId,
      },
    });
    Logger.debug("Linked author to paper", { authorId, paperId });
  }
}

/**
 * Detects arXiv URLs in text content and returns the first one found
 *
 * @param content - The text content to search for arXiv URLs
 * @returns The first arXiv URL found, or null if none found
 */
export function detectArxivUrl(content: string): string | null {
  if (!content) return null;

  // Look for arXiv URLs in the content
  const arxivUrlPattern = /https?:\/\/arxiv\.org\/abs\/\d+\.\d+/gi;
  const matches = content.match(arxivUrlPattern);

  return matches ? matches[0] : null;
}

/**
 * Auto-imports a paper from arXiv
 *
 * @param arxivUrl - The arXiv URL to import
 * @returns The paper ID if successfully imported/found, or null if failed
 */
export async function autoImportArxivPaper(
  arxivUrl: string
): Promise<string | null> {
  try {
    Logger.info("Auto-importing arXiv paper", { arxivUrl });

    // Extract arXiv ID from URL
    const arxivId = extractArxivId(arxivUrl);

    // Check if paper already exists in database
    const existingPaper = await prisma.paper.findFirst({
      where: {
        OR: [{ externalId: arxivId }, { link: arxivUrl }],
      },
    });

    if (existingPaper) {
      Logger.info("Paper already exists", {
        paperId: existingPaper.id,
        title: existingPaper.title,
      });

      // Check if existing paper needs LLM abstract generation
      if (!existingPaper.llmAbstract) {
        Logger.info("Queuing LLM abstract generation for existing paper", {
          paperId: existingPaper.id,
        });
        try {
          const { queueLLMAbstractGeneration } = await import(
            "./background-jobs"
          );
          await queueLLMAbstractGeneration(existingPaper.id);
        } catch (error) {
          Logger.error(
            "Failed to queue LLM abstract for existing paper",
            error instanceof Error ? error : new Error(String(error)),
            { paperId: existingPaper.id }
          );
        }
      }

      return existingPaper.id;
    }

    // Fetch paper data from arXiv API
    Logger.info("Fetching paper data from arXiv API", { arxivId });
    const paperData = await fetchArxivPaper(arxivId);

    // Create paper record in database
    const paper = await prisma.paper.create({
      data: {
        title: paperData.title,
        abstract: paperData.abstract,
        link: paperData.arxivUrl,
        source: "arxiv",
        externalId: paperData.id,
        tags: [], // No longer importing arXiv categories as tags
      },
    });

    // Create authors and link them to the paper
    Logger.info("Processing paper authors", {
      paperId: paper.id,
      authorCount: paperData.authors.length,
    });
    for (const authorName of paperData.authors) {
      const author = await getOrCreateAuthor(authorName);
      await linkAuthorToPaper(author.id, paper.id);
    }

    Logger.info("Successfully imported paper", {
      paperId: paper.id,
      title: paper.title,
    });

    // Generate LLM abstract in the background using job queue
    Logger.info("Queuing LLM abstract generation for paper", {
      paperId: paper.id,
    });
    try {
      const { queueLLMAbstractGeneration } = await import("./background-jobs");
      await queueLLMAbstractGeneration(paper.id);
    } catch (error) {
      Logger.error(
        "Failed to queue LLM abstract for paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId: paper.id }
      );
    }

    // Auto-tag the paper in the background using job queue
    Logger.info("Queuing auto-tagging for paper", { paperId: paper.id });
    try {
      const { queueAutoTagging } = await import("./background-jobs");
      await queueAutoTagging(paper.id);
    } catch (error) {
      Logger.error(
        "Failed to queue auto-tagging for paper",
        error instanceof Error ? error : new Error(String(error)),
        { paperId: paper.id }
      );
    }

    return paper.id;
  } catch (error) {
    Logger.error(
      "Failed to auto-import arXiv paper",
      error instanceof Error ? error : new Error(String(error)),
      { arxivUrl }
    );
    return null;
  }
}

/**
 * Enhances an existing paper with institution data from PDF
 *
 * @param paperId - The paper ID to enhance
 * @param arxivUrl - The arXiv URL for PDF fetching
 */
export async function enhanceArxivPaperWithInstitutions(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  try {
    Logger.info("Enhancing paper with institution data", { paperId });

    // Get paper data for authors list
    const paper = await prisma.paper.findUnique({
      where: { id: paperId },
      include: {
        paperAuthors: {
          include: { author: true },
        },
      },
    });

    if (!paper) {
      Logger.warn("Paper not found for institution enhancement", { paperId });
      return;
    }

    const authorNames = paper.paperAuthors.map((pa) => pa.author.name);

    // Construct PDF URL directly from stored arXiv ID (no API call needed)
    const pdfUrl = `https://arxiv.org/pdf/${paper.externalId}.pdf`;

    let institutions: string[] = [];
    try {
      Logger.info("Fetching PDF for institution extraction", { pdfUrl });
      const pdfResponse = await fetch(pdfUrl);

      if (pdfResponse.ok) {
        const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
        Logger.info("Extracting institutions from PDF", { paperId });
        institutions = await extractInstitutionsFromPdf(pdfBuffer, authorNames);

        // Filter out obvious noise and keep only reasonable institutions
        institutions = institutions.filter((inst) => {
          const cleaned = inst.trim();
          return (
            cleaned.length > 4 &&
            cleaned.length < 60 &&
            !cleaned.includes("al.,") &&
            !cleaned.includes("limited") &&
            !cleaned.includes("et") &&
            /university|institute|college|tech|corp|inc|lab|center|centre/i.test(
              cleaned
            )
          );
        });

        Logger.info("Found institutions in PDF", {
          paperId,
          institutionCount: institutions.length,
          institutions,
        });
      } else {
        Logger.warn("Could not fetch PDF for institution extraction", {
          paperId,
          pdfUrl,
        });
      }
    } catch (error) {
      Logger.error(
        "Error extracting institutions",
        error instanceof Error ? error : new Error(String(error)),
        { paperId }
      );
    }

    // Create Institution entities and link to paper
    for (const institutionName of institutions) {
      try {
        // Find or create Institution entity
        const institution = await prisma.institution.upsert({
          where: { name: institutionName },
          create: {
            name: institutionName,
            type: "UNIVERSITY", // Default type, can be updated manually later
          },
          update: {}, // No updates needed if it exists
        });

        // Create PaperInstitution join record (if not already exists)
        await prisma.paperInstitution.upsert({
          where: {
            paperId_institutionId: {
              paperId: paperId,
              institutionId: institution.id,
            },
          },
          create: {
            paperId: paperId,
            institutionId: institution.id,
          },
          update: {}, // No updates needed if link already exists
        });

        Logger.debug("Linked institution to paper", {
          institutionName,
          paperId,
        });
      } catch (error) {
        Logger.error(
          "Failed to link institution",
          error instanceof Error ? error : new Error(String(error)),
          { institutionName, paperId }
        );
      }
    }

    Logger.info("Enhanced paper with institutions", {
      paperId,
      institutionCount: institutions.length,
    });
  } catch (error) {
    Logger.error(
      "Failed to enhance paper with institutions",
      error instanceof Error ? error : new Error(String(error)),
      { paperId }
    );
  }
}

/**
 * Processes post content for arXiv URLs and auto-imports papers (synchronous version)
 *
 * @param content - The post content to process
 * @returns The paper ID if an arXiv paper was found and imported, or null
 */
export async function processArxivAutoImport(
  content: string
): Promise<string | null> {
  const arxivUrl = detectArxivUrl(content);

  if (!arxivUrl) {
    return null;
  }

  return await autoImportArxivPaper(arxivUrl); // Use worker version for background processing
}

/**
 * Async version that enhances paper with institutions in the background
 *
 * Uses database data to avoid redundant API calls and downloads.
 *
 * @param paperId - The paper ID to enhance with institutions
 * @param arxivUrl - The arXiv URL (for compatibility, not used - PDF URL constructed from DB)
 */
export async function processArxivInstitutionsAsync(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  Logger.info("Background institution processing for paper", { paperId });

  try {
    await enhanceArxivPaperWithInstitutions(paperId, arxivUrl);
    Logger.info("Background institution processing complete", { paperId });
  } catch (error) {
    Logger.error(
      "Background institution processing failed",
      error instanceof Error ? error : new Error(String(error)),
      { paperId }
    );
  }
}

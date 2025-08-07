import { prisma } from "@/lib/db/prisma";
import {
  extractArxivId,
  fetchArxivPaper,
} from "../../scripts/fetch-arxiv-paper";
import { extractInstitutionsFromPdf } from "./pdf-institution-extractor";

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
        orcid: null,
        googleScholarId: null,
        arxivId: null,
      },
    });
    console.log(`  ✅ Created author: ${normalizedName}`);
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
    console.log(`  🔗 Linked author to paper`);
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
 * Auto-imports a paper from arXiv (fast, without institutions)
 *
 * @param arxivUrl - The arXiv URL to import
 * @returns The paper ID if successfully imported/found, or null if failed
 */
export async function autoImportArxivPaperSync(
  arxivUrl: string
): Promise<string | null> {
  try {
    console.log(`🔍 Auto-importing arXiv paper (sync): ${arxivUrl}`);

    // Extract arXiv ID from URL
    const arxivId = extractArxivId(arxivUrl);

    // Check if paper already exists in database
    const existingPaper = await prisma.paper.findFirst({
      where: {
        OR: [{ externalId: arxivId }, { link: arxivUrl }],
      },
    });

    if (existingPaper) {
      console.log(`✅ Paper already exists: ${existingPaper.title}`);
      return existingPaper.id;
    }

    // Fetch paper data from arXiv API
    console.log(`📡 Fetching paper data from arXiv API...`);
    const paperData = await fetchArxivPaper(arxivId);

    // Create paper record in database WITHOUT institutions (fast)
    const paper = await prisma.paper.create({
      data: {
        title: paperData.title,
        abstract: paperData.abstract,
        link: paperData.arxivUrl,
        source: "arxiv",
        externalId: paperData.id,
        tags: paperData.categories, // Use arXiv categories as tags
        institutions: [], // Empty for now - will be filled by background process
      },
    });

    // Create authors and link them to the paper
    console.log(`👥 Processing ${paperData.authors.length} authors...`);
    for (const authorName of paperData.authors) {
      const author = await getOrCreateAuthor(authorName);
      await linkAuthorToPaper(author.id, paper.id);
    }

    console.log(`✅ Successfully imported paper (sync): ${paper.title}`);
    return paper.id;
  } catch (error) {
    console.error(`❌ Failed to auto-import arXiv paper (sync):`, error);
    return null;
  }
}

/**
 * Auto-imports a paper from arXiv (full version with institutions)
 *
 * @param arxivUrl - The arXiv URL to import
 * @returns The paper ID if successfully imported/found, or null if failed
 */
export async function autoImportArxivPaper(
  arxivUrl: string
): Promise<string | null> {
  try {
    console.log(`🔍 Auto-importing arXiv paper: ${arxivUrl}`);

    // Extract arXiv ID from URL
    const arxivId = extractArxivId(arxivUrl);

    // Check if paper already exists in database
    const existingPaper = await prisma.paper.findFirst({
      where: {
        OR: [{ externalId: arxivId }, { link: arxivUrl }],
      },
    });

    if (existingPaper) {
      console.log(`✅ Paper already exists: ${existingPaper.title}`);
      return existingPaper.id;
    }

    // Fetch paper data from arXiv API
    console.log(`📡 Fetching paper data from arXiv API...`);
    const paperData = await fetchArxivPaper(arxivId);

    // Fetch PDF and extract institutions
    let institutions: string[] = [];
    try {
      console.log(`📄 Fetching PDF for institution extraction...`);
      const pdfUrl = paperData.pdfUrl;
      const pdfResponse = await fetch(pdfUrl);

      if (pdfResponse.ok) {
        const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
        console.log(`🔍 Extracting institutions from PDF...`);
        institutions = await extractInstitutionsFromPdf(
          pdfBuffer,
          paperData.authors
        );

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

        console.log(
          `🏛️ Found ${institutions.length} institutions: ${institutions.join(", ")}`
        );
      } else {
        console.log(`⚠️ Could not fetch PDF for institution extraction`);
      }
    } catch (error) {
      console.error(`❌ Error extracting institutions:`, error);
    }

    // Create paper record in database
    const paper = await prisma.paper.create({
      data: {
        title: paperData.title,
        abstract: paperData.abstract,
        link: paperData.arxivUrl,
        source: "arxiv",
        externalId: paperData.id,
        tags: paperData.categories, // Use arXiv categories as tags
        institutions: institutions, // Institutions extracted from PDF
      },
    });

    // Create authors and link them to the paper
    console.log(`👥 Processing ${paperData.authors.length} authors...`);
    for (const authorName of paperData.authors) {
      const author = await getOrCreateAuthor(authorName);
      await linkAuthorToPaper(author.id, paper.id);
    }

    console.log(`✅ Successfully imported paper: ${paper.title}`);
    return paper.id;
  } catch (error) {
    console.error(`❌ Failed to auto-import arXiv paper:`, error);
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
    console.log(`🏛️ Enhancing paper ${paperId} with institution data...`);

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
      console.error(`❌ Paper ${paperId} not found`);
      return;
    }

    const authorNames = paper.paperAuthors.map((pa) => pa.author.name);

    // Extract arXiv ID and fetch PDF
    const arxivId = extractArxivId(arxivUrl);
    const paperData = await fetchArxivPaper(arxivId);

    let institutions: string[] = [];
    try {
      console.log(`📄 Fetching PDF for institution extraction...`);
      const pdfUrl = paperData.pdfUrl;
      const pdfResponse = await fetch(pdfUrl);

      if (pdfResponse.ok) {
        const pdfBuffer = Buffer.from(await pdfResponse.arrayBuffer());
        console.log(`🔍 Extracting institutions from PDF...`);
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

        console.log(
          `🏛️ Found ${institutions.length} institutions: ${institutions.join(", ")}`
        );
      } else {
        console.log(`⚠️ Could not fetch PDF for institution extraction`);
      }
    } catch (error) {
      console.error(`❌ Error extracting institutions:`, error);
    }

    // Update paper with institutions
    await prisma.paper.update({
      where: { id: paperId },
      data: {
        institutions: institutions,
      },
    });

    console.log(`✅ Enhanced paper ${paperId} with institutions`);
  } catch (error) {
    console.error(`❌ Failed to enhance paper with institutions:`, error);
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

  return await autoImportArxivPaperSync(arxivUrl); // Use sync version for immediate import
}

/**
 * Async version that enhances paper with institutions in the background
 *
 * @param paperId - The paper ID to enhance with institutions
 * @param arxivUrl - The arXiv URL for PDF fetching
 */
export async function processArxivInstitutionsAsync(
  paperId: string,
  arxivUrl: string
): Promise<void> {
  console.log(`🚀 Background institution processing for paper: ${paperId}`);

  try {
    await enhanceArxivPaperWithInstitutions(paperId, arxivUrl);
    console.log(
      `✅ Background institution processing complete for paper ${paperId}`
    );
  } catch (error) {
    console.error(
      `❌ Background institution processing failed for paper ${paperId}:`,
      error
    );
  }
}

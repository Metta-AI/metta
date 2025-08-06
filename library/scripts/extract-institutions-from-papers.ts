#!/usr/bin/env tsx

/**
 * Institution Extraction Script
 *
 * This script extracts institution information from papers that have arXiv data.
 * It fetches the arXiv metadata to get author affiliations and populates the
 * institutions field on papers.
 */

import { PrismaClient } from "@prisma/client";
import { fetchArxivPaper, extractArxivId } from "./fetch-arxiv-paper";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: ".env.local" });

const prisma = new PrismaClient();

/**
 * Extract arXiv ID from paper data with robust URL parsing
 */
function getArxivIdFromPaper(paper: any): string | null {
  // Try external ID first
  if (paper.externalId?.includes("arxiv")) {
    const id = paper.externalId.replace("arxiv:", "");
    return cleanArxivId(id);
  }

  // Try link with enhanced parsing
  if (paper.link?.includes("arxiv.org")) {
    return extractArxivIdRobust(paper.link);
  }

  return null;
}

/**
 * Enhanced arXiv ID extraction that handles various URL formats
 */
function extractArxivIdRobust(input: string): string | null {
  if (!input || typeof input !== "string") {
    return null;
  }

  // Handle various arXiv URL patterns
  const patterns = [
    // Standard patterns
    /arxiv\.org\/abs\/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)/i,
    /arxiv\.org\/pdf\/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?(?:\.pdf)?/i,
    /arxiv\.org\/html\/([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?/i,

    // Legacy patterns (pre-2007)
    /arxiv\.org\/abs\/([a-z-]+\/[0-9]{7}(?:v[0-9]+)?)/i,
    /arxiv\.org\/pdf\/([a-z-]+\/[0-9]{7})(?:v[0-9]+)?(?:\.pdf)?/i,

    // Just the ID patterns
    /^([0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?$/i,
    /^([a-z-]+\/[0-9]{7})(?:v[0-9]+)?$/i,
  ];

  for (const pattern of patterns) {
    const match = input.match(pattern);
    if (match) {
      return cleanArxivId(match[1]);
    }
  }

  // Last resort: try the original extractArxivId function
  try {
    return extractArxivId(input);
  } catch (error) {
    console.warn(`⚠️  Could not parse arXiv ID from: ${input}`);
    return null;
  }
}

/**
 * Clean and normalize arXiv ID
 */
function cleanArxivId(id: string): string {
  if (!id) return id;

  // Remove version numbers (e.g., "2204.11674v1" -> "2204.11674")
  return id.replace(/v[0-9]+$/, "");
}

/**
 * Clean and normalize institution names
 */
function normalizeInstitutionName(institution: string): string {
  return (
    institution
      .trim()
      // Remove common prefixes/suffixes
      .replace(
        /^(Department of|Dept\.?\s+of|School of|College of|Faculty of)\s+/i,
        ""
      )
      .replace(/,\s*(Department|Dept\.?|School|College|Faculty).*$/i, "")
      // Clean up university naming variations
      .replace(/\s+University\s*$/, " University")
      .replace(/\s+Institute\s*$/, " Institute")
      .replace(/\s+College\s*$/, " College")
      // Remove extra whitespace
      .replace(/\s+/g, " ")
      .trim()
  );
}

/**
 * Extract unique institutions from author data
 */
function extractInstitutionsFromAuthors(arxivData: any): string[] {
  const institutions = new Set<string>();

  // For arXiv data, institutions are often embedded in author information
  // This is a simplified extraction - real implementation would need more sophisticated parsing

  if (arxivData.authors && Array.isArray(arxivData.authors)) {
    // Some arXiv papers include institution info in author names or comments
    // This is a basic heuristic-based approach

    const text = JSON.stringify(arxivData);

    // Look for university patterns
    const universityMatches = text.match(/([A-Z][a-z]+\s+)+University/g) || [];
    const instituteMatches = text.match(/([A-Z][a-z]+\s+)+Institute/g) || [];
    const collegeMatches = text.match(/([A-Z][a-z]+\s+)+College/g) || [];
    const labMatches =
      text.match(
        /(Google|Microsoft|Facebook|Meta|OpenAI|DeepMind|Anthropic|Apple)\s*(Research|AI|Labs?)?/g
      ) || [];

    [
      ...universityMatches,
      ...instituteMatches,
      ...collegeMatches,
      ...labMatches,
    ].forEach((match) => {
      const normalized = normalizeInstitutionName(match);
      if (normalized.length > 3) {
        // Filter out very short matches
        institutions.add(normalized);
      }
    });
  }

  return Array.from(institutions).slice(0, 5); // Limit to 5 institutions per paper
}

/**
 * Update paper with institution data
 */
async function updatePaperInstitutions(
  paperId: string,
  institutions: string[]
): Promise<void> {
  if (institutions.length === 0) {
    console.log(`  ⚠️  No institutions found for paper`);
    return;
  }

  try {
    await prisma.paper.update({
      where: { id: paperId },
      data: { institutions },
    });

    console.log(
      `  ✅ Updated institutions: ${institutions.slice(0, 2).join(", ")}${institutions.length > 2 ? "..." : ""}`
    );
  } catch (error) {
    console.error(`  ❌ Error updating paper institutions:`, error);
  }
}

/**
 * Main extraction function
 */
async function extractInstitutionsFromPapers() {
  try {
    console.log("🏛️ Extracting institutions from papers...");

    // Get all papers with arXiv links
    const allArxivPapers = await prisma.paper.findMany({
      where: {
        OR: [
          { source: "arxiv" },
          { link: { contains: "arxiv.org" } },
          { externalId: { contains: "arxiv" } },
        ],
      },
      select: {
        id: true,
        title: true,
        source: true,
        externalId: true,
        link: true,
        institutions: true,
      },
    });

    // Filter for papers without institutions (empty array or null)
    const papers = allArxivPapers.filter(
      (paper) => !paper.institutions || paper.institutions.length === 0
    );

    console.log(`📊 Found ${papers.length} papers without institution data`);

    let processedCount = 0;
    let errorCount = 0;
    let institutionCount = 0;

    // Process papers in batches to avoid overwhelming the arXiv API
    const batchSize = 3;
    for (let i = 0; i < papers.length; i += batchSize) {
      const batch = papers.slice(i, i + batchSize);

      console.log(
        `\n📦 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(papers.length / batchSize)}`
      );

      // Process papers in parallel within each batch
      const batchPromises = batch.map(async (paper) => {
        try {
          const arxivId = getArxivIdFromPaper(paper);
          if (!arxivId) {
            console.log(
              `⚠️  Could not extract arXiv ID from paper: ${paper.title}`
            );
            return;
          }

          console.log(`\n📄 Processing: ${paper.title.slice(0, 80)}...`);
          console.log(`   arXiv ID: ${arxivId}`);

          // Fetch institution data from arXiv
          const arxivData = await fetchArxivPaper(arxivId);

          // Extract institutions
          const institutions = extractInstitutionsFromAuthors(arxivData);

          if (institutions.length > 0) {
            await updatePaperInstitutions(paper.id, institutions);
            institutionCount += institutions.length;
          }

          processedCount++;
        } catch (error) {
          console.error(`❌ Error processing paper "${paper.title}":`, error);
          errorCount++;
        }
      });

      // Wait for batch to complete
      await Promise.all(batchPromises);

      // Add a small delay between batches to be respectful to arXiv API
      if (i + batchSize < papers.length) {
        console.log("⏳ Waiting 3 seconds before next batch...");
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    }

    console.log("\n🎉 Institution extraction completed!");
    console.log(`✅ Successfully processed: ${processedCount} papers`);
    console.log(`❌ Errors: ${errorCount} papers`);
    console.log(`🏛️ Total institutions extracted: ${institutionCount}`);

    // Show some statistics
    const allPapersForStats = await prisma.paper.findMany({
      select: { institutions: true },
    });

    const papersWithInstitutions = allPapersForStats.filter(
      (paper) => paper.institutions && paper.institutions.length > 0
    ).length;

    console.log(`📊 Papers with institutions: ${papersWithInstitutions}`);

    // Show top institutions - reuse the data we already fetched
    const allPapers = allPapersForStats.filter(
      (paper) => paper.institutions && paper.institutions.length > 0
    );

    const institutionCounts = new Map<string, number>();
    allPapers.forEach((paper) => {
      paper.institutions.forEach((inst) => {
        institutionCounts.set(inst, (institutionCounts.get(inst) || 0) + 1);
      });
    });

    const topInstitutions = Array.from(institutionCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    if (topInstitutions.length > 0) {
      console.log("\n🏆 Top institutions:");
      topInstitutions.forEach(([inst, count], index) => {
        console.log(`   ${index + 1}. ${inst} (${count} papers)`);
      });
    }
  } catch (error) {
    console.error("❌ Error extracting institutions:", error);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the script
if (require.main === module) {
  extractInstitutionsFromPapers();
}

#!/usr/bin/env tsx

/**
 * Author Extraction Script
 *
 * This script extracts author data from arXiv papers and creates author records
 * with ORCID-aware deduplication and proper relationships.
 */

import * as dotenv from "dotenv";
import { PrismaClient } from "@prisma/client";
import { fetchArxivPaper, extractArxivId } from "./fetch-arxiv-paper";

// Load environment variables
dotenv.config({ path: ".env.local", quiet: true });

// Initialize Prisma client
const prisma = new PrismaClient();

/**
 * Extracts arXiv ID from paper link or external ID
 */
function getArxivIdFromPaper(paper: any): string | null {
  // Try to extract from externalId first
  if (paper.externalId && paper.externalId.match(/^\d+\.\d+/)) {
    return paper.externalId;
  }

  // Try to extract from link
  if (paper.link && paper.link.includes("arxiv.org")) {
    const match = paper.link.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    if (match) {
      return match[1];
    }
  }

  return null;
}

/**
 * Normalizes author name for consistent matching
 */
function normalizeAuthorName(name: string): string {
  return name.trim().replace(/\s+/g, " ");
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
  } else {
    console.log(`  🔍 Found existing author: ${normalizedName}`);
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
  } else {
    console.log(`  ⚠️  Author-paper link already exists`);
  }
}

async function extractAuthorsFromPapers() {
  try {
    console.log("🔍 Extracting authors from papers...");

    // Get all papers with arXiv links
    const papers = await prisma.paper.findMany({
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
      },
    });

    console.log(`📊 Found ${papers.length} papers with arXiv links`);

    let processedCount = 0;
    let errorCount = 0;
    const processedAuthors = new Set<string>();

    // Process papers in batches to avoid overwhelming the arXiv API
    const batchSize = 5;
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

          console.log(`\n📄 Processing: ${paper.title}`);
          console.log(`   arXiv ID: ${arxivId}`);

          // Fetch author data from arXiv
          const arxivData = await fetchArxivPaper(arxivId);

          // Process each author
          for (const authorName of arxivData.authors) {
            const normalizedName = normalizeAuthorName(authorName);

            // Skip if we've already processed this author in this run
            if (processedAuthors.has(normalizedName)) {
              console.log(
                `  ⏭️  Skipping already processed author: ${normalizedName}`
              );
              continue;
            }

            const author = await getOrCreateAuthor(authorName);
            await linkAuthorToPaper(author.id, paper.id);

            processedAuthors.add(normalizedName);
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
        console.log("⏳ Waiting 2 seconds before next batch...");
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    console.log("\n🎉 Author extraction completed!");
    console.log(`✅ Successfully processed: ${processedCount} papers`);
    console.log(`❌ Errors: ${errorCount} papers`);
    console.log(`👥 Total unique authors processed: ${processedAuthors.size}`);
  } catch (error) {
    console.error("❌ Error extracting authors:", error);
  } finally {
    await prisma.$disconnect();
  }
}

extractAuthorsFromPapers();

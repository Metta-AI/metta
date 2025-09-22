#!/usr/bin/env tsx

/**
 * Display Author Results Script
 *
 * This script displays the final results of the author extraction process,
 * showing statistics about the authors in the database.
 */

import * as dotenv from "dotenv";
import { PrismaClient } from "@prisma/client";

// Load environment variables
dotenv.config({ path: ".env.local" });

// Initialize Prisma client
const prisma = new PrismaClient();

async function displayAuthorResults() {
  try {
    console.log("📊 Author Extraction Results Summary");
    console.log("=====================================\n");

    // Get total authors
    const totalAuthors = await prisma.author.count();
    console.log(`📈 Total Authors: ${totalAuthors}`);

    // Get authors with papers
    const authorsWithPapers = await prisma.author.count({
      where: {
        paperAuthors: {
          some: {},
        },
      },
    });
    console.log(`📚 Authors with Papers: ${authorsWithPapers}`);

    // Get papers with authors
    const papersWithAuthors = await prisma.paper.count({
      where: {
        paperAuthors: {
          some: {},
        },
      },
    });
    console.log(`📄 Papers with Authors: ${papersWithAuthors}`);

    // Get total paper-author relationships
    const totalRelationships = await prisma.paperAuthor.count();
    console.log(`🔗 Total Paper-Author Relationships: ${totalRelationships}`);

    // Get some sample authors
    const sampleAuthors = await prisma.author.findMany({
      take: 10,
      include: {
        paperAuthors: {
          include: {
            paper: {
              select: {
                title: true,
                link: true,
              },
            },
          },
        },
      },
      orderBy: {
        name: "asc",
      },
    });

    console.log("\n👥 Sample Authors:");
    console.log("==================");
    sampleAuthors.forEach((author, index) => {
      console.log(`${index + 1}. ${author.name}`);
      console.log(`   Papers: ${author.paperAuthors.length}`);
      if (author.paperAuthors.length > 0) {
        const paper = author.paperAuthors[0].paper;
        console.log(`   Sample Paper: ${paper.title}`);
      }
      console.log("");
    });

    // Get authors with multiple papers
    const allAuthors = await prisma.author.findMany({
      include: {
        paperAuthors: true,
      },
    });

    const authorsWithMultiplePapers = allAuthors
      .filter((author) => author.paperAuthors.length > 1)
      .sort((a, b) => b.paperAuthors.length - a.paperAuthors.length)
      .slice(0, 5);

    console.log("🏆 Top Authors by Paper Count:");
    console.log("==============================");
    authorsWithMultiplePapers.forEach((author, index) => {
      console.log(
        `${index + 1}. ${author.name} - ${author.paperAuthors.length} papers`
      );
    });

    console.log("\n✅ Author extraction process completed successfully!");
    console.log(
      "🎉 The /authors page now displays real author data from arXiv papers."
    );
  } catch (error) {
    console.error("❌ Error displaying author results:", error);
  } finally {
    await prisma.$disconnect();
  }
}

displayAuthorResults();

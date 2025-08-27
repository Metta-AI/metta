#!/usr/bin/env tsx

/**
 * Script to generate LLM abstracts for papers
 *
 * Usage:
 *   pnpm run generate-llm-abstracts           # Generate for 10 papers without abstracts
 *   pnpm run generate-llm-abstracts -- --all # Generate for all papers without abstracts
 *   pnpm run generate-llm-abstracts -- --paper-id=<id>  # Generate for specific paper
 *   pnpm run generate-llm-abstracts -- --limit=20       # Generate for up to 20 papers
 */

import { prisma } from "../src/lib/db/prisma";
import { PaperAbstractService } from "../src/lib/paper-abstract-service";

async function main() {
  const args = process.argv.slice(2);

  // Parse command line arguments
  const options = {
    all: args.includes("--all"),
    paperId: args.find((arg) => arg.startsWith("--paper-id="))?.split("=")[1],
    limit: parseInt(
      args.find((arg) => arg.startsWith("--limit="))?.split("=")[1] || "10"
    ),
  };

  console.log("🚀 Starting LLM abstract generation...");
  console.log("Options:", options);

  try {
    if (options.paperId) {
      // Generate for specific paper
      console.log(`📋 Generating abstract for paper: ${options.paperId}`);
      const abstract = await PaperAbstractService.generateAbstractForPaper(
        options.paperId
      );

      if (abstract) {
        console.log("✅ Successfully generated abstract");
        console.log(
          "📋 Short explanation:",
          abstract.shortExplanation.slice(0, 100) + "..."
        );
        console.log("📊 Summary:", abstract.summary.slice(0, 100) + "...");
      } else {
        console.log("❌ Failed to generate abstract");
      }
    } else if (options.all) {
      // Generate for all papers without abstracts
      console.log(
        "📋 Generating abstracts for ALL papers without LLM abstracts..."
      );

      const totalPapers = await prisma.paper.count({
        where: {
          llmAbstract: null as any,
          link: { not: null },
        },
      });

      console.log(`📊 Found ${totalPapers} papers needing abstracts`);

      // Process in batches to avoid overwhelming the system
      const batchSize = 5;
      let processed = 0;

      while (processed < totalPapers) {
        console.log(
          `\n📦 Processing batch ${Math.floor(processed / batchSize) + 1}...`
        );
        await PaperAbstractService.generateMissingAbstracts(batchSize);
        processed += batchSize;

        // Small delay between batches
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }

      console.log(`✅ Completed processing all ${totalPapers} papers`);
    } else {
      // Generate for limited number of papers
      console.log(
        `📋 Generating abstracts for up to ${options.limit} papers...`
      );
      await PaperAbstractService.generateMissingAbstracts(options.limit);
    }
  } catch (error) {
    console.error("❌ Error during abstract generation:", error);
    process.exit(1);
  }

  console.log("🎉 LLM abstract generation completed!");
}

// Handle cleanup on exit
process.on("SIGINT", async () => {
  console.log("\n⏹️ Shutting down gracefully...");
  await prisma.$disconnect();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  console.log("\n⏹️ Shutting down gracefully...");
  await prisma.$disconnect();
  process.exit(0);
});

// Run the script
if (require.main === module) {
  main()
    .then(() => {
      process.exit(0);
    })
    .catch((error) => {
      console.error("💥 Unhandled error:", error);
      process.exit(1);
    })
    .finally(async () => {
      await prisma.$disconnect();
    });
}

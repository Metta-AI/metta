#!/usr/bin/env tsx

/**
 * Database Reset and Full Import Script
 *
 * This script resets the database to a fresh state and runs all import scripts:
 * 1. Resets the database (clears all data)
 * 2. Imports papers from Asana
 * 3. Extracts authors from papers
 * 4. Enriches author metadata
 * 5. Extracts institutions from papers
 * 6. Shows final database contents
 *
 * Usage:
 *   tsx scripts/reset-and-import-all.ts
 *   tsx scripts/reset-and-import-all.ts --skip-reset  # Skip database reset
 *   tsx scripts/reset-and-import-all.ts --dry-run     # Show what would be done
 */

import { exec } from "child_process";
import { promisify } from "util";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: ".env.local", quiet: true });

const execAsync = promisify(exec);

interface ScriptStep {
  name: string;
  command: string;
  description: string;
  required: boolean;
}

const RESET_STEPS: ScriptStep[] = [
  {
    name: "db-reset",
    command: "pnpm db:reset --force",
    description: "Reset database to fresh state (clears all data)",
    required: true,
  },
  {
    name: "db-push",
    command: "pnpm db:push",
    description: "Apply database schema to create tables",
    required: true,
  },
];

const IMPORT_STEPS: ScriptStep[] = [
  {
    name: "asana-import",
    command: "pnpm pull-asana-papers",
    description: "Import papers from Asana project",
    required: true,
  },
  {
    name: "extract-authors",
    command: "pnpm extract-authors",
    description: "Extract authors from imported papers",
    required: true,
  },
  {
    name: "enrich-metadata",
    command: "pnpm enrich-metadata",
    description: "Enrich author metadata with additional information",
    required: false,
  },
  {
    name: "extract-institutions",
    command: "pnpm extract-institutions",
    description: "Extract institutions from papers",
    required: true,
  },
];

const VERIFICATION_STEPS: ScriptStep[] = [
  {
    name: "show-contents",
    command: "tsx scripts/show-database-contents.ts",
    description: "Display final database contents summary",
    required: false,
  },
];

async function runCommand(step: ScriptStep): Promise<boolean> {
  console.log(`\n🔄 ${step.description}...`);
  console.log(`   Command: ${step.command}`);

  try {
    const { stdout, stderr } = await execAsync(step.command);

    if (stdout) {
      console.log("📄 Output:");
      console.log(stdout);
    }

    if (stderr && !stderr.includes("warn") && !stderr.includes("WARN")) {
      console.log("⚠️ Warnings/Errors:");
      console.log(stderr);
    }

    console.log(`✅ ${step.name} completed successfully`);
    return true;
  } catch (error: any) {
    console.error(`❌ ${step.name} failed:`, error.message);

    if (step.required) {
      console.error(`💥 Required step failed. Stopping execution.`);
      process.exit(1);
    } else {
      console.log(`⏭️ Optional step failed, continuing...`);
      return false;
    }
  }
}

async function checkPrerequisites(): Promise<boolean> {
  console.log("🔍 Checking prerequisites...");

  // Check for required environment variables
  const requiredEnvVars = ["DATABASE_URL"];
  const optionalEnvVars = ["ASANA_API_KEY", "ANTHROPIC_API_KEY"];

  let allRequired = true;

  for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
      console.error(`❌ Required environment variable ${envVar} is not set`);
      allRequired = false;
    } else {
      console.log(`✅ ${envVar} is set`);
    }
  }

  for (const envVar of optionalEnvVars) {
    if (!process.env[envVar]) {
      console.log(`⚠️ Optional environment variable ${envVar} is not set`);
    } else {
      console.log(`✅ ${envVar} is set`);
    }
  }

  if (!allRequired) {
    console.error(
      "💥 Missing required environment variables. Please check your .env.local file."
    );
    return false;
  }

  console.log("✅ Prerequisites check passed");
  return true;
}

async function main() {
  const args = process.argv.slice(2);
  const skipReset = args.includes("--skip-reset");
  const dryRun = args.includes("--dry-run");

  console.log("🚀 Database Reset and Full Import Script");
  console.log("=".repeat(50));

  if (dryRun) {
    console.log("🔍 DRY RUN MODE - No actual changes will be made\n");
  }

  // Check prerequisites
  if (!(await checkPrerequisites())) {
    process.exit(1);
  }

  const allSteps = [
    ...(skipReset ? [] : RESET_STEPS),
    ...IMPORT_STEPS,
    ...VERIFICATION_STEPS,
  ];

  console.log("\n📋 Planned execution steps:");
  allSteps.forEach((step, index) => {
    const status = step.required ? "(required)" : "(optional)";
    console.log(`   ${index + 1}. ${step.description} ${status}`);
  });

  if (dryRun) {
    console.log("\n✅ Dry run completed. Use without --dry-run to execute.");
    process.exit(0);
  }

  // Confirm execution for reset
  if (!skipReset) {
    console.log("\n⚠️ WARNING: This will DELETE ALL DATABASE DATA!");
    console.log("📊 Current database will be completely reset.");

    // In a real scenario, you might want to prompt for confirmation
    // For script usage, we'll continue automatically
    console.log("🔄 Proceeding with database reset in 3 seconds...");
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }

  const startTime = Date.now();
  let successCount = 0;
  let failCount = 0;

  // Execute all steps
  for (const [index, step] of allSteps.entries()) {
    console.log(`\n📍 Step ${index + 1}/${allSteps.length}: ${step.name}`);

    const success = await runCommand(step);
    if (success) {
      successCount++;
    } else {
      failCount++;
    }
  }

  const endTime = Date.now();
  const duration = Math.round((endTime - startTime) / 1000);

  console.log("\n" + "=".repeat(50));
  console.log("📊 EXECUTION SUMMARY");
  console.log("=".repeat(50));
  console.log(`⏱️ Total time: ${duration} seconds`);
  console.log(`✅ Successful steps: ${successCount}`);
  console.log(`❌ Failed steps: ${failCount}`);
  console.log(`📋 Total steps: ${allSteps.length}`);

  if (failCount === 0) {
    console.log("\n🎉 All steps completed successfully!");
    console.log("📚 Your library database is now fully populated with:");
    console.log("   • Papers imported from Asana");
    console.log("   • Author information extracted and enriched");
    console.log("   • Institution affiliations extracted");
    console.log("\n🚀 Ready to use! Try running the dev server with: pnpm dev");
  } else {
    console.log(`\n⚠️ Completed with ${failCount} failed steps.`);
    console.log("💡 Check the logs above for details on failed steps.");
    if (failCount < allSteps.length) {
      console.log(
        "✅ The database should still be functional with partial data."
      );
    }
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error("💥 Script execution failed:", error);
    process.exit(1);
  });
}

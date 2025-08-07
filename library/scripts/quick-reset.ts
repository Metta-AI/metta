#!/usr/bin/env tsx

/**
 * Quick Database Reset Script
 *
 * A simplified version of the full reset script for development.
 * Only resets the database and imports papers from Asana.
 *
 * Usage:
 *   tsx scripts/quick-reset.ts
 */

import { exec } from "child_process";
import { promisify } from "util";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: ".env.local" });

const execAsync = promisify(exec);

async function runCommand(name: string, command: string): Promise<boolean> {
  console.log(`\n🔄 ${name}...`);
  console.log(`   Running: ${command}`);

  try {
    const { stdout, stderr } = await execAsync(command);

    if (stdout) {
      // Only show important output, filter out verbose logs
      const lines = stdout
        .split("\n")
        .filter(
          (line) =>
            line.includes("✅") ||
            line.includes("❌") ||
            line.includes("imported") ||
            line.includes("created") ||
            line.includes("found") ||
            line.includes("error")
        );
      if (lines.length > 0) {
        console.log("📄 Key output:");
        lines.slice(0, 10).forEach((line) => console.log(`   ${line}`));
      }
    }

    console.log(`✅ ${name} completed`);
    return true;
  } catch (error: any) {
    console.error(`❌ ${name} failed:`, error.message);
    return false;
  }
}

async function main() {
  console.log("⚡ Quick Database Reset");
  console.log("=".repeat(30));

  const startTime = Date.now();

  // Step 1: Reset database
  const resetSuccess = await runCommand(
    "Database Reset",
    "pnpm db:reset --force"
  );

  if (!resetSuccess) {
    console.error("💥 Database reset failed. Exiting.");
    process.exit(1);
  }

  // Step 2: Apply database schema
  const schemaSuccess = await runCommand("Apply Schema", "pnpm db:push");

  if (!schemaSuccess) {
    console.error("💥 Database schema push failed. Exiting.");
    process.exit(1);
  }

  // Step 3: Import papers from Asana
  const asanaSuccess = await runCommand(
    "Asana Import",
    "pnpm pull-asana-papers"
  );

  if (!asanaSuccess) {
    console.log("⚠️ Asana import failed, but database is reset");
  }

  const endTime = Date.now();
  const duration = Math.round((endTime - startTime) / 1000);

  console.log("\n" + "=".repeat(30));
  console.log(`⏱️ Completed in ${duration} seconds`);

  if (resetSuccess && schemaSuccess && asanaSuccess) {
    console.log("🎉 Quick reset successful!");
    console.log(
      '💡 Run "pnpm reset-and-import" for full import with authors & institutions'
    );
  } else if (resetSuccess && schemaSuccess) {
    console.log("✅ Database reset and schema applied successfully");
    console.log("⚠️ Asana import had issues - check your ASANA_API_KEY");
  } else if (resetSuccess) {
    console.log("✅ Database reset successful");
    console.log("❌ Schema application failed");
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error("💥 Quick reset failed:", error);
    process.exit(1);
  });
}

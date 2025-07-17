#!/usr/bin/env tsx

/**
 * Papers Verification Script
 * 
 * This script verifies the imported papers data and displays a sample
 * for review without needing a web frontend.
 */

import { db } from "../src/lib/db";
import { papersTable, userPaperInteractionsTable } from "../src/lib/db/schema/paper";
import { usersTable } from "../src/lib/db/schema/auth";
import { eq } from "drizzle-orm";

/**
 * Display paper information in a readable format
 */
function displayPaper(paper: any, index: number) {
  console.log(`\n📄 Paper ${index + 1}:`);
  console.log(`   ID: ${paper.id}`);
  console.log(`   Title: ${paper.title}`);
  console.log(`   Source: ${paper.source || 'manual'}`);
  console.log(`   External ID: ${paper.externalId || 'N/A'}`);
  console.log(`   Tags: ${paper.tags?.join(', ') || 'None'}`);
  console.log(`   Stars: ${paper.stars}`);
  console.log(`   Starred: ${paper.starred ? 'Yes' : 'No'}`);
  console.log(`   Link: ${paper.link || 'N/A'}`);
  console.log(`   Created: ${paper.createdAt}`);
}

/**
 * Display user interaction information
 */
function displayUserInteraction(interaction: any, user: any) {
  console.log(`   👤 ${user.name} (${user.email}):`);
  console.log(`      - Starred: ${interaction.starred ? 'Yes' : 'No'}`);
  console.log(`      - Read: ${interaction.readAt ? interaction.readAt : 'No'}`);
  console.log(`      - Queued: ${interaction.queued ? 'Yes' : 'No'}`);
  if (interaction.notes) {
    console.log(`      - Notes: ${interaction.notes}`);
  }
}

/**
 * Main verification function
 */
async function verifyPapers() {
  console.log('🔍 Verifying imported papers data...\n');

  try {
    // Get total counts
    const paperCount = await db.select({ count: papersTable.id }).from(papersTable);
    const interactionCount = await db.select({ count: userPaperInteractionsTable.userId }).from(userPaperInteractionsTable);
    const userCount = await db.select({ count: usersTable.id }).from(usersTable);

    console.log('📊 Database Summary:');
    console.log(`   Papers: ${paperCount.length}`);
    console.log(`   User Interactions: ${interactionCount.length}`);
    console.log(`   Users: ${userCount.length}`);

    // Get sample papers
    console.log('\n📋 Sample Papers (first 5):');
    const samplePapers = await db.select().from(papersTable).limit(5);
    
    if (samplePapers.length === 0) {
      console.log('❌ No papers found in database');
      return;
    }

    samplePapers.forEach((paper, index) => {
      displayPaper(paper, index);
    });

    // Get user interactions for the first paper
    if (samplePapers.length > 0) {
      const firstPaperId = samplePapers[0].id;
      console.log(`\n🔗 User Interactions for "${samplePapers[0].title}":`);
      
      const interactions = await db
        .select({
          interaction: userPaperInteractionsTable,
          user: usersTable
        })
        .from(userPaperInteractionsTable)
        .innerJoin(usersTable, eq(userPaperInteractionsTable.userId, usersTable.id))
        .where(eq(userPaperInteractionsTable.paperId, firstPaperId));

      if (interactions.length === 0) {
        console.log('   No user interactions found');
      } else {
        interactions.forEach(({ interaction, user }) => {
          displayUserInteraction(interaction, user);
        });
      }
    }

    // Show some statistics
    console.log('\n📈 Statistics:');
    
    // Papers by source
    const papersBySource = await db
      .select({ source: papersTable.source, count: papersTable.id })
      .from(papersTable);
    
    const sourceCounts = papersBySource.reduce((acc, row) => {
      const source = row.source || 'manual';
      acc[source] = (acc[source] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    console.log('   Papers by source:');
    Object.entries(sourceCounts).forEach(([source, count]) => {
      console.log(`     ${source}: ${count}`);
    });

    // Starred papers
    const starredPapers = await db
      .select()
      .from(papersTable)
      .where(eq(papersTable.starred, true));
    
    console.log(`   Starred papers: ${starredPapers.length}`);

    // Papers with tags (count papers that have non-empty tags arrays)
    const allPapers = await db.select().from(papersTable);
    const papersWithTags = allPapers.filter(paper => paper.tags && paper.tags.length > 0);
    
    console.log(`   Papers with tags: ${papersWithTags.length}`);

    console.log('\n✅ Verification complete!');

  } catch (error) {
    console.error('❌ Error during verification:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    process.exit(1);
  }
}

// Run verification
verifyPapers().catch(console.error); 
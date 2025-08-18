#!/usr/bin/env tsx

/**
 * Author Metadata Enrichment Script
 * 
 * This script enriches author metadata by calculating statistics from their
 * existing papers in the database, without adding new papers.
 * 
 * Usage:
 *   tsx scripts/enrich-author-metadata.ts
 *   tsx scripts/enrich-author-metadata.ts --dry-run
 */

import * as dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

// Initialize Prisma client
const prisma = new PrismaClient();

/**
 * Interface for tracking enrichment progress
 */
interface EnrichmentProgress {
  totalAuthors: number;
  processedAuthors: number;
  successfulAuthors: number;
  failedAuthors: number;
  metadataUpdated: number;
  errors: string[];
}

/**
 * Calculates author statistics from their existing papers
 */
async function calculateAuthorStats(authorId: string): Promise<{
  paperCount: number;
  totalStars: number;
  avgStars: number;
  categories: string[];
  institutions: string[];
  recentActivity: Date | null;
}> {
  // Get all papers for this author
  const authorPapers = await prisma.paperAuthor.findMany({
    where: { authorId },
    include: {
      paper: {
        select: {
          stars: true,
          tags: true,
          institutions: true,
          updatedAt: true
        }
      }
    }
  });

  if (authorPapers.length === 0) {
    return {
      paperCount: 0,
      totalStars: 0,
      avgStars: 0,
      categories: [],
      institutions: [],
      recentActivity: null
    };
  }

  // Calculate statistics
  const paperCount = authorPapers.length;
  const totalStars = authorPapers.reduce((sum, pa) => sum + pa.paper.stars, 0);
  const avgStars = totalStars / paperCount;

  // Collect unique categories and institutions
  const categories = new Set<string>();
  const institutions = new Set<string>();

  authorPapers.forEach(pa => {
    pa.paper.tags.forEach(tag => categories.add(tag));
    pa.paper.institutions.forEach(inst => institutions.add(inst));
  });

  // Find most recent activity
  const recentActivity = authorPapers.reduce((latest, pa) => {
    return pa.paper.updatedAt > latest ? pa.paper.updatedAt : latest;
  }, authorPapers[0].paper.updatedAt);

  return {
    paperCount,
    totalStars,
    avgStars,
    categories: Array.from(categories),
    institutions: Array.from(institutions),
    recentActivity
  };
}

/**
 * Updates author metadata in the database
 */
async function updateAuthorMetadata(
  authorId: string, 
  stats: {
    paperCount: number;
    totalStars: number;
    avgStars: number;
    categories: string[];
    institutions: string[];
    recentActivity: Date | null;
  }
): Promise<void> {
  // Determine expertise areas based on categories
  const expertise = stats.categories.slice(0, 5); // Top 5 categories
  
  // Determine primary institution (most common)
  const primaryInstitution = stats.institutions.length > 0 ? stats.institutions[0] : null;
  
  // Update author record
  await prisma.author.update({
    where: { id: authorId },
    data: {
      institution: primaryInstitution,
      hIndex: stats.paperCount, // Simple approximation
      totalCitations: Math.round(stats.totalStars), // Using stars as citation proxy
      recentActivity: stats.recentActivity,
      updatedAt: new Date()
    }
  });
  
  // Update expertise separately to avoid type issues
  if (expertise.length > 0) {
    await prisma.author.update({
      where: { id: authorId },
      data: {
        expertise: expertise
      }
    });
  }
}

/**
 * Processes metadata for a single author
 */
async function processAuthorMetadata(
  author: { id: string; name: string },
  progress: EnrichmentProgress
): Promise<void> {
  try {
    console.log(`\n🔍 Processing metadata for: ${author.name}`);
    
    // Calculate statistics from existing papers
    const stats = await calculateAuthorStats(author.id);
    
    if (stats.paperCount === 0) {
      console.log(`  ℹ️  No papers found for ${author.name}`);
      progress.processedAuthors++;
      return;
    }
    
    console.log(`  📊 Found ${stats.paperCount} papers with ${stats.totalStars} total stars`);
    console.log(`  📈 Categories: ${stats.categories.slice(0, 3).join(', ')}...`);
    console.log(`  🏢 Institutions: ${stats.institutions.slice(0, 2).join(', ')}...`);
    
    // Update author metadata
    await updateAuthorMetadata(author.id, stats);
    
    console.log(`  ✅ Updated metadata for ${author.name}`);
    
    progress.metadataUpdated++;
    progress.successfulAuthors++;
    
  } catch (error) {
    console.error(`  ❌ Error processing metadata for ${author.name}:`, error);
    progress.errors.push(`Author ${author.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    progress.failedAuthors++;
  } finally {
    progress.processedAuthors++;
  }
}

/**
 * Main enrichment function
 */
async function enrichAuthorMetadata(dryRun: boolean = false): Promise<void> {
  console.log('📚 Author Metadata Enrichment Script');
  console.log('='.repeat(50));
  console.log(`Dry run: ${dryRun ? 'Yes' : 'No'}`);
  console.log('');
  
  // Initialize progress tracking
  const progress: EnrichmentProgress = {
    totalAuthors: 0,
    processedAuthors: 0,
    successfulAuthors: 0,
    failedAuthors: 0,
    metadataUpdated: 0,
    errors: []
  };
  
  try {
    // Get all authors from database
    const authors = await prisma.author.findMany({
      orderBy: { name: 'asc' }
    });
    
    progress.totalAuthors = authors.length;
    console.log(`📊 Found ${authors.length} authors in database`);
    
    if (dryRun) {
      console.log('🔍 DRY RUN - No changes will be made');
      console.log('Authors that would be processed:');
      authors.slice(0, 10).forEach((author, index) => {
        console.log(`  ${index + 1}. ${author.name}`);
      });
      console.log(`  ... and ${authors.length - 10} more`);
      return;
    }
    
    // Process each author
    for (let i = 0; i < authors.length; i++) {
      const author = authors[i];
      
      console.log(`\n[${i + 1}/${authors.length}] Processing: ${author.name}`);
      
      await processAuthorMetadata(author, progress);
      
      // Add small delay to avoid overwhelming the database
      if (i < authors.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
    
    // Print final summary
    console.log('\n' + '='.repeat(50));
    console.log('📊 METADATA ENRICHMENT SUMMARY');
    console.log('='.repeat(50));
    console.log(`Total authors: ${progress.totalAuthors}`);
    console.log(`Processed: ${progress.processedAuthors}`);
    console.log(`Successful: ${progress.successfulAuthors}`);
    console.log(`Failed: ${progress.failedAuthors}`);
    console.log(`Metadata updated: ${progress.metadataUpdated}`);
    
    if (progress.errors.length > 0) {
      console.log(`\n❌ Errors encountered: ${progress.errors.length}`);
      progress.errors.slice(0, 5).forEach((error, index) => {
        console.log(`  ${index + 1}. ${error}`);
      });
      if (progress.errors.length > 5) {
        console.log(`  ... and ${progress.errors.length - 5} more errors`);
      }
    }
    
  } catch (error) {
    console.error('❌ Fatal error:', error);
    throw error;
  }
}

/**
 * Main function for command-line usage
 */
async function main() {
  const args = process.argv.slice(2);
  
  let dryRun = false;
  
  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--dry-run') {
      dryRun = true;
    }
  }
  
  try {
    await enrichAuthorMetadata(dryRun);
  } catch (error) {
    console.error(`❌ Script failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

// Run main function if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error(`❌ Script failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    process.exit(1);
  });
} 
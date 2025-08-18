#!/usr/bin/env tsx

/**
 * Papers Import Script
 * 
 * This script imports papers data from the mockup into the new database schema.
 * It handles the transformation from the mock data structure to the new schema.
 */

import { PrismaClient } from '@prisma/client';
import * as fs from 'fs';
import * as path from 'path';

// Initialize Prisma client
const prisma = new PrismaClient();

// Import the papers data from the metta-library-mock directory
const papersDataPath = path.join(__dirname, '..', '..', '..', 'metta-library-mock', 'observatory', 'src', 'mockData', 'papers.ts');

interface MockUser {
  id: string;
  name: string;
  avatar: string;
  email: string;
}

interface MockPaper {
  id: string;
  title: string;
  starred: boolean;
  authors: string[]; // Array of scholar IDs (empty in current data)
  institutions: string[]; // Array of institution IDs (empty in current data)
  tags: string[];
  readBy: MockUser[];
  queued: MockUser[];
  link: string;
  stars: number;
}

/**
 * Load papers data from the mockup file
 */
function loadPapersData(): MockPaper[] {
  if (!fs.existsSync(papersDataPath)) {
    throw new Error(`Papers data file not found: ${papersDataPath}`);
  }

  const fileContent = fs.readFileSync(papersDataPath, 'utf8');
  
  // Extract the mockPapers array from the file content
  const papersMatch = fileContent.match(/export const mockPapers: Paper\[\] = (\[[\s\S]*?\]);/);
  
  if (!papersMatch) {
    throw new Error('Could not find mockPapers array in papers.ts');
  }
  
  const papersArrayString = papersMatch[1];
  
  // The data might have trailing commas, so we need to clean it up
  const cleanedString = papersArrayString
    .replace(/,(\s*[}\]])/g, '$1') // Remove trailing commas
    .replace(/\n/g, ' ') // Replace newlines with spaces
    .replace(/\s+/g, ' '); // Normalize whitespace
  
  return JSON.parse(cleanedString);
}

/**
 * Load users data - we'll create a simple test user since we don't need mock users
 */
function loadUsersData(): MockUser[] {
  // Return empty array - we'll create a test user in the database if needed
  return [];
}

/**
 * Transform mock paper to database schema
 */
function transformPaper(mockPaper: MockPaper) {
  // Extract source from link
  let source = 'manual';
  let externalId = '';
  
  if (mockPaper.link.includes('arxiv.org')) {
    source = 'arxiv';
    // Extract arXiv ID from URL
    const arxivMatch = mockPaper.link.match(/arxiv\.org\/abs\/(\d+\.\d+)/);
    if (arxivMatch) {
      externalId = arxivMatch[1];
    }
  } else if (mockPaper.link.includes('biorxiv.org')) {
    source = 'biorxiv';
  }

  return {
    id: mockPaper.id,
    title: mockPaper.title,
    abstract: null, // Not available in mock data
    authors: mockPaper.authors, // Empty array for now
    institutions: mockPaper.institutions, // Empty array for now
    tags: mockPaper.tags,
    link: mockPaper.link,
    source,
    externalId,
    stars: mockPaper.stars,
    starred: mockPaper.starred,
    pdfS3Url: null, // Future use
    createdAt: new Date(),
    updatedAt: new Date()
  };
}

/**
 * Create user paper interactions
 */
function createUserInteractions(mockPaper: MockPaper) {
  const interactionsMap = new Map<string, any>();

  // Add readBy interactions
  for (const user of mockPaper.readBy) {
    const key = `${user.id}-${mockPaper.id}`;
    interactionsMap.set(key, {
      userId: user.id,
      paperId: mockPaper.id,
      starred: false,
      readAt: new Date(), // Use current time as read time
      queued: false,
      notes: null
    });
  }

  // Add queued interactions (will override readBy if user appears in both)
  for (const user of mockPaper.queued) {
    const key = `${user.id}-${mockPaper.id}`;
    interactionsMap.set(key, {
      userId: user.id,
      paperId: mockPaper.id,
      starred: false,
      readAt: null,
      queued: true,
      notes: null
    });
  }

  return Array.from(interactionsMap.values());
}

/**
 * Main import function
 */
async function importPapers() {
  console.log('📄 Starting papers import...\n');

  try {
    // Load papers data from file
    console.log('📂 Loading papers data from mockup file...');
    const papers = loadPapersData();
    
    console.log(`   Loaded ${papers.length} papers`);

    // Create users
    console.log('👥 Creating users...');
    
    // Extract unique users from the mock data
    const uniqueUsers = new Set<string>();
    for (const paper of papers) {
      for (const user of paper.readBy) {
        uniqueUsers.add(user.id);
      }
      for (const user of paper.queued) {
        uniqueUsers.add(user.id);
      }
    }
    
    // Create user records
    const userRecords = Array.from(uniqueUsers).map(userId => ({
      id: userId,
      name: userId.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), // Convert kebab-case to Title Case
      email: `${userId}@example.com`,
      image: null
    }));
    
    // Check if the specific users we need already exist
    const existingUserIds = await prisma.user.findMany({ select: { id: true } });
    const existingUserSet = new Set(existingUserIds.map(u => u.id));
    
    // Filter out users that already exist
    const newUsers = userRecords.filter(user => !existingUserSet.has(user.id));
    
    if (newUsers.length > 0) {
      await prisma.user.createMany({ data: newUsers });
      console.log(`✅ Created ${newUsers.length} new users`);
    } else {
      console.log('ℹ️  All required users already exist');
    }

    // Transform papers
    console.log(`📋 Processing ${papers.length} papers...`);
    
    const transformedPapers = papers.map(transformPaper);
    
    // Check if papers already exist
    const existingPapers = await prisma.paper.findMany({ take: 1 });
    
    if (existingPapers.length === 0) {
      // Insert papers only if they don't exist
      console.log('📋 Inserting papers...');
      await prisma.paper.createMany({ data: transformedPapers });
      console.log('✅ Papers imported successfully');
    } else {
      console.log('ℹ️  Papers already exist, skipping insertion');
    }

    // Create user interactions
    console.log('🔗 Creating user interactions...');
    const allInteractions = papers.flatMap(createUserInteractions);
    
    if (allInteractions.length > 0) {
      await prisma.userPaperInteraction.createMany({ data: allInteractions });
      console.log(`✅ Created ${allInteractions.length} user interactions`);
    } else {
      console.log('ℹ️  No user interactions to create');
    }

    console.log('\n🎉 Import completed successfully!');
    console.log(`📊 Summary:`);
    console.log(`  - Papers imported: ${transformedPapers.length}`);
    console.log(`  - User interactions created: ${allInteractions.length}`);

  } catch (error) {
    console.error('❌ Error during import:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    process.exit(1);
  }
}

// Run import
importPapers().catch(console.error); 
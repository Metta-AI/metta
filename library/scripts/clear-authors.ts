#!/usr/bin/env tsx

/**
 * Clear Authors Data Script
 * 
 * This script clears existing author data for testing
 */

import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function clearAuthors() {
  try {
    console.log('Clearing all authors from database...');

    // Delete all paper-author relationships first
    await prisma.paperAuthor.deleteMany();
    console.log('✅ Deleted all paper-author relationships');

    // Delete all authors
    const result = await prisma.author.deleteMany();
    console.log(`✅ Deleted ${result.count} authors`);

    console.log('✅ Authors cleared successfully!');
  } catch (error) {
    console.error('Error clearing authors:', error);
  } finally {
    await prisma.$disconnect();
  }
}

clearAuthors(); 
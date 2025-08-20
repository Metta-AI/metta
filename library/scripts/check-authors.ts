#!/usr/bin/env tsx

/**
 * Temporary script to examine current author data in papers table
 */

import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function checkAuthors() {
  try {
    console.log('Checking authors in database...');

    const authors = await prisma.author.findMany({
      include: {
        paperAuthors: {
          include: {
            paper: true
          }
        }
      }
    });

    console.log(`Found ${authors.length} authors:`);
    
    authors.forEach(author => {
      console.log(`- ${author.name} (${author.paperAuthors.length} papers)`);
    });

  } catch (error) {
    console.error('Error checking authors:', error);
  } finally {
    await prisma.$disconnect();
  }
}

checkAuthors(); 
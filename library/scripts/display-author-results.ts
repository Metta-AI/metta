#!/usr/bin/env tsx

/**
 * Display Author Results Script
 * 
 * This script displays the results of author extraction in a clear format
 */

import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function displayAuthorResults() {
  try {
    console.log('Displaying author extraction results...');

    const authors = await prisma.author.findMany({
      include: {
        paperAuthors: {
          include: {
            paper: {
              select: {
                id: true,
                title: true,
                source: true,
                externalId: true
              }
            }
          }
        }
      },
      orderBy: {
        name: 'asc'
      }
    });

    console.log(`\n📊 Found ${authors.length} authors:\n`);

    authors.forEach(author => {
      console.log(`👤 ${author.name}`);
      console.log(`   ID: ${author.id}`);
      console.log(`   Papers: ${author.paperAuthors.length}`);
      
      if (author.paperAuthors.length > 0) {
        console.log('   Paper titles:');
        author.paperAuthors.forEach(pa => {
          const paper = pa.paper;
          console.log(`     - ${paper.title} (${paper.source || 'unknown'})`);
        });
      }
      console.log('');
    });

  } catch (error) {
    console.error('Error displaying author results:', error);
  } finally {
    await prisma.$disconnect();
  }
}

displayAuthorResults(); 
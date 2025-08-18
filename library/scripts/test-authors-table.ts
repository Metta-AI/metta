#!/usr/bin/env tsx

/**
 * Test script to verify the new Authors table functionality
 */

import { PrismaClient } from '@prisma/client';

// Initialize Prisma client
const prisma = new PrismaClient();

async function testAuthorsTable() {
  try {
    console.log('Testing authors table...');

    // Test creating an author
    const testAuthor = await prisma.author.create({
      data: {
        name: 'Test Author',
        email: 'test@example.com',
        institution: 'Test University',
        orcid: null,
        googleScholarId: null,
        arxivId: null
      }
    });

    console.log('✅ Created test author:', testAuthor);

    // Test finding the author
    const foundAuthor = await prisma.author.findUnique({
      where: { id: testAuthor.id }
    });

    console.log('✅ Found author:', foundAuthor);

    // Test updating the author
    const updatedAuthor = await prisma.author.update({
      where: { id: testAuthor.id },
      data: {
        institution: 'Updated University'
      }
    });

    console.log('✅ Updated author:', updatedAuthor);

    // Test deleting the author
    await prisma.author.delete({
      where: { id: testAuthor.id }
    });

    console.log('✅ Deleted test author');

    // Test counting all authors
    const authorCount = await prisma.author.count();
    console.log('✅ Total authors in database:', authorCount);

    console.log('🎉 All tests passed!');

  } catch (error) {
    console.error('❌ Test failed:', error);
  } finally {
    await prisma.$disconnect();
  }
}

testAuthorsTable(); 
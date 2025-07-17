#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { db } from "../src/lib/db";
import { papersTable } from "../src/lib/db/schema/paper";
import { eq } from 'drizzle-orm';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

async function testDatabase() {
  try {
    console.log('🔍 Testing database connection...');
    
    // Test basic connection
    const result = await db.select().from(papersTable).limit(1);
    console.log('✅ Database connection successful');
    console.log(`   Found ${result.length} papers in table`);
    
    // Test inserting a single paper
    console.log('📝 Testing paper insertion...');
    const testPaper = {
      id: 'test-paper-1',
      title: 'Test Paper',
      abstract: 'This is a test paper',
      authors: ['Test Author'],
      institutions: ['Test University'],
      tags: ['test'],
      link: 'https://example.com',
      source: 'test',
      externalId: 'test-123',
      stars: 0,
      starred: false,
      pdf_s3_url: null,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    await db.insert(papersTable).values(testPaper);
    console.log('✅ Paper insertion successful');
    
    // Verify the paper was inserted
    const insertedPaper = await db.select().from(papersTable).where(eq(papersTable.id, 'test-paper-1'));
    console.log(`✅ Found inserted paper: ${insertedPaper[0]?.title}`);
    
    // Clean up
    await db.delete(papersTable).where(eq(papersTable.id, 'test-paper-1'));
    console.log('✅ Test paper cleaned up');
    
  } catch (error) {
    console.error('❌ Database test failed:', error);
    throw error;
  }
}

testDatabase().catch(console.error); 
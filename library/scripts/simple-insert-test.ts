#!/usr/bin/env tsx

import * as dotenv from 'dotenv';

// Load environment variables FIRST
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

// Then import db and schemas
import { db } from "../src/lib/db";
import { papersTable } from "../src/lib/db/schema/paper";
import { eq } from 'drizzle-orm';

async function simpleInsertTest() {
  console.log('🔍 Testing simple paper insertion...');
  
  try {
    // Try to insert a single test paper
    const testPaper = {
      id: 'test-paper-1',
      title: 'Test Paper',
      abstract: null,
      authors: [],
      institutions: [],
      tags: ['test'],
      link: 'https://example.com',
      source: 'manual',
      externalId: '',
      stars: 0,
      starred: false,
      pdfS3Url: null,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    console.log('📝 Attempting to insert test paper...');
    await db.insert(papersTable).values(testPaper);
    console.log('✅ Test paper inserted successfully!');
    
    // Verify it was inserted
    const result = await db.select().from(papersTable).where(eq(papersTable.id, 'test-paper-1'));
    console.log('✅ Paper found in database:', result);
    
  } catch (error) {
    console.error('❌ Simple insert test failed:', error);
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
  }
}

simpleInsertTest().catch(console.error); 
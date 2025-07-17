#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { db } from "../src/lib/db";

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

import { papersTable, userPaperInteractionsTable } from "../src/lib/db/schema/paper";
import { usersTable } from "../src/lib/db/schema/auth";

async function testImportImports() {
  console.log('🔍 Testing with import script imports...');
  
  try {
    // Test with papersTable from the schema
    const result = await db.select().from(papersTable).limit(1);
    console.log('✅ Import script imports work:', result);
    
  } catch (error) {
    console.error('❌ Import script imports failed:', error);
  }
}

testImportImports().catch(console.error); 
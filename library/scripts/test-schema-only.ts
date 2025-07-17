#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';
import { papersTable } from '../src/lib/db/schema/paper';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

async function testSchemaOnly() {
  console.log('🔍 Testing Drizzle with schema...');
  
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
  });
  
  // Create drizzle instance with ONLY the paper schema
  const db = drizzle(pool, {
    schema: { papersTable }
  });
  
  try {
    // Test schema-based query
    const result = await db.select().from(papersTable).limit(1);
    console.log('✅ Schema-based query works:', result);
    
  } catch (error) {
    console.error('❌ Schema test failed:', error);
  } finally {
    await pool.end();
  }
}

testSchemaOnly().catch(console.error); 
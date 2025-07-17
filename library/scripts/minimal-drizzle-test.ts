#!/usr/bin/env tsx

import * as dotenv from 'dotenv';
import { drizzle } from 'drizzle-orm/node-postgres';
import { Pool } from 'pg';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local', override: true });

async function minimalTest() {
  console.log('🔍 Minimal Drizzle test...');
  console.log(`   DATABASE_URL: ${process.env.DATABASE_URL}`);
  
  // Create a simple pool
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
  });
  
  // Create drizzle instance with NO schema
  const db = drizzle(pool);
  
  try {
    // Test a simple query without any schema
    const result = await db.execute('SELECT 1 as test');
    console.log('✅ Basic Drizzle query works:', result);
    
    // Test a simple table query
    const tableResult = await db.execute('SELECT COUNT(*) FROM paper');
    console.log('✅ Direct table query works:', tableResult);
    
  } catch (error) {
    console.error('❌ Drizzle test failed:', error);
  } finally {
    await pool.end();
  }
}

minimalTest().catch(console.error); 